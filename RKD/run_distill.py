import os
import argparse
import socket
import csv

import dataset
import model.backbone as backbone
import metric.pairsampler as pair

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader

from metric.utils import recall, pdist
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, RkdQuadrupletSum
from model.embedding import LinearEmbedding


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset',
                    choices=dict(cub200=dataset.CUB2011Metric,
                                 cars196=dataset.Cars196Metric,
                                 stanford=dataset.StanfordOnlineProductsMetric),
                    default=dataset.CUB2011Metric,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(googlenet=backbone.GoogleNet,
                                 inception_v1bn=backbone.InceptionV1BN,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--teacher_base',
                    choices=dict(googlenet=backbone.GoogleNet,
                                 inception_v1bn=backbone.InceptionV1BN,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--triplet_ratio', default=0, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--angle_ratio', default=0, type=float)
parser.add_argument('--quad_ratio', default=0, type=float)

parser.add_argument('--dark_ratio', default=0, type=float)
parser.add_argument('--dark_alpha', default=2, type=float)
parser.add_argument('--dark_beta', default=3, type=float)

parser.add_argument('--at_ratio', default=0, type=float)

parser.add_argument('--triplet_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.DistanceWeighted,
                    action=LookupChoices)

parser.add_argument('--triplet_margin', type=float, default=0.2)
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
parser.add_argument('--embedding_size', default=128, type=int)

parser.add_argument('--teacher_load', default=None, required=True)
parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')
parser.add_argument('--teacher_embedding_size', default=128, type=int)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--save_dir', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--recall', default=[1, 2, 4, 8], type=int, nargs='+')

parser.add_argument('--log_confusion_matrix', choices=['true', 'false'], default='true')
parser.add_argument('--max_confusion_classes', type=int, default=200)
parser.add_argument('--max_confusion_samples', type=int, default=2000)

parser.add_argument('--wandb_project', default='rkd-metric-learning')
parser.add_argument('--wandb_entity', default=None)
parser.add_argument('--wandb_run_name', default=None)
parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
parser.add_argument('--wandb_group', default='distillation-experiments')

opts = parser.parse_args()
student_base = opts.base(pretrained=True)
teacher_base = opts.teacher_base(pretrained=False)


def _name(x):
    return x.__name__ if hasattr(x, '__name__') else str(x)


def get_normalize(net):
    google_mean = torch.Tensor([104, 117, 128]).view(1, -1, 1, 1).cuda()
    google_std = torch.Tensor([1, 1, 1]).view(1, -1, 1, 1).cuda()
    other_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
    other_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()

    def googlenorm(x):
        x = x[:, [2, 1, 0]] * 255
        x = (x - google_mean) / google_std
        return x

    def othernorm(x):
        x = (x - other_mean) / other_std
        return x

    if isinstance(net, backbone.InceptionV1BN) or isinstance(net, backbone.GoogleNet):
        return googlenorm
    else:
        return othernorm


teacher_normalize = get_normalize(teacher_base)
student_normalize = get_normalize(student_base)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=True)
dataset_train_eval = opts.dataset(opts.data, train=True, transform=test_transform, download=True)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=True)

print("Number of images in Training Set: %d" % len(dataset_train))
print("Number of images in Test set: %d" % len(dataset_eval))

run = wandb.init(
    project=opts.wandb_project,
    entity=opts.wandb_entity,
    name=opts.wandb_run_name,
    mode=opts.wandb_mode,
    group=opts.wandb_group,
    config={
        'dataset': _name(opts.dataset),
        'base': _name(opts.base),
        'teacher_base': _name(opts.teacher_base),
        'triplet_ratio': opts.triplet_ratio,
        'dist_ratio': opts.dist_ratio,
        'angle_ratio': opts.angle_ratio,
        'quad_ratio': opts.quad_ratio,
        'dark_ratio': opts.dark_ratio,
        'dark_alpha': opts.dark_alpha,
        'dark_beta': opts.dark_beta,
        'at_ratio': opts.at_ratio,
        'triplet_sample': _name(opts.triplet_sample),
        'triplet_margin': opts.triplet_margin,
        'l2normalize': opts.l2normalize,
        'embedding_size': opts.embedding_size,
        'teacher_load': opts.teacher_load,
        'teacher_l2normalize': opts.teacher_l2normalize,
        'teacher_embedding_size': opts.teacher_embedding_size,
        'lr': opts.lr,
        'data': opts.data,
        'epochs': opts.epochs,
        'batch': opts.batch,
        'iter_per_epoch': opts.iter_per_epoch,
        'lr_decay_epochs': opts.lr_decay_epochs,
        'lr_decay_gamma': opts.lr_decay_gamma,
        'recall': opts.recall,
        'log_confusion_matrix': opts.log_confusion_matrix,
        'max_confusion_classes': opts.max_confusion_classes,
        'max_confusion_samples': opts.max_confusion_samples,
        'save_dir': opts.save_dir,
        'load': opts.load,
        'wandb_group': opts.wandb_group,
    },
    tags=['rkd', 'metric-learning', 'distillation', 'run_distill.py'],
)

run.summary['hostname'] = socket.gethostname()
run.summary['train_size'] = len(dataset_train)
run.summary['test_size'] = len(dataset_eval)

dataset_artifact = wandb.Artifact(
    name='dataset-%s-distill' % _name(opts.dataset).lower(),
    type='dataset',
    metadata={
        'dataset_class': _name(opts.dataset),
        'data_dir': os.path.abspath(opts.data),
        'train_size': len(dataset_train),
        'test_size': len(dataset_eval),
    },
)
run.log_artifact(dataset_artifact)

loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5,
                                                                     iter_per_epoch=opts.iter_per_epoch),
                                 pin_memory=True, num_workers=8)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=8)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=8)

student = LinearEmbedding(student_base,
                          output_size=student_base.output_size,
                          embedding_size=opts.embedding_size,
                          normalize=opts.l2normalize == 'true')

if opts.load is not None:
    student.load_state_dict(torch.load(opts.load))
    print("Loaded Model from %s" % opts.load)

teacher = LinearEmbedding(teacher_base,
                          output_size=teacher_base.output_size,
                          embedding_size=opts.teacher_embedding_size,
                          normalize=opts.teacher_l2normalize == 'true')

teacher.load_state_dict(torch.load(opts.teacher_load))
student = student.cuda()
teacher = teacher.cuda()

optimizer = optim.Adam(student.parameters(), lr=opts.lr, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
triplet_criterion = L2Triplet(sampler=opts.triplet_sample(), margin=opts.triplet_margin)
at_criterion = AttentionTransfer()
quad_criterion = RkdQuadrupletSum()
primary_recall_k = opts.recall[0]

history_columns = [
    'epoch',
    'train_loss',
    'train_triplet_loss',
    'train_dist_loss',
    'train_angle_loss',
    'train_quad_loss',
    'train_dark_loss',
    'train_at_loss',
    'lr',
]
for k in opts.recall:
    history_columns.extend([
        'train_recall%d' % k,
        'test_recall%d' % k,
        'best_train_recall%d' % k,
        'best_test_recall%d' % k,
    ])
history_table = wandb.Table(columns=history_columns)


def _recall_log_dict(prefix, rec_values):
    out = {}
    for k, r in zip(opts.recall, rec_values):
        out['%s_recall%d' % (prefix, k)] = r
    return out


def _append_history_row(epoch, train_stats, train_rec, test_rec, best_train_rec, best_test_rec, lr):
    row = [
        epoch,
        train_stats['loss'] if train_stats is not None else None,
        train_stats['triplet'] if train_stats is not None else None,
        train_stats['dist'] if train_stats is not None else None,
        train_stats['angle'] if train_stats is not None else None,
        train_stats['quad'] if train_stats is not None else None,
        train_stats['dark'] if train_stats is not None else None,
        train_stats['at'] if train_stats is not None else None,
        lr,
    ]
    for i, _ in enumerate(opts.recall):
        row.extend([
            train_rec[i],
            test_rec[i],
            best_train_rec[i],
            best_test_rec[i],
        ])
    history_table.add_data(*row)


def _log_confusion_matrix(prefix, embeddings, labels, step):
    if opts.log_confusion_matrix == 'false':
        return

    if embeddings.size(0) < 2:
        return

    if embeddings.size(0) > opts.max_confusion_samples:
        idx = torch.randperm(embeddings.size(0))[:opts.max_confusion_samples]
        embeddings = embeddings[idx]
        labels = labels[idx]

    d = pdist(embeddings, squared=True)
    d.fill_diagonal_(float('inf'))
    nn_idx = d.argmin(dim=1)

    y_true_raw = labels.tolist()
    y_pred_raw = labels[nn_idx].tolist()

    classes = sorted(set(y_true_raw) | set(y_pred_raw))
    if len(classes) > opts.max_confusion_classes:
        run.log({
            '%s/confusion_matrix_skipped' % prefix: 1,
            '%s/confusion_num_classes' % prefix: len(classes),
        }, step=step)
        return

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = [class_to_idx[c] for c in y_true_raw]
    y_pred = [class_to_idx[c] for c in y_pred_raw]
    class_names = [str(c) for c in classes]

    run.log({
        '%s/confusion_matrix' % prefix: wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )
    }, step=step)


def train(loader, ep):
    lr_scheduler.step()
    student.train()
    teacher.eval()

    dist_loss_all = []
    angle_loss_all = []
    dark_loss_all = []
    quad_loss_all = []
    triplet_loss_all = []
    at_loss_all = []
    loss_all = []

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, t_pool, t_e = teacher(teacher_normalize(images), True)

        if isinstance(student.base, backbone.GoogleNet):
            assert (opts.at_ratio == 0), "AttentionTransfer cannot be applied on GoogleNet at current implementation."
            e = student(student_normalize(images))
            at_loss = torch.zeros(1, device=e.device)
        else:
            b1, b2, b3, b4, pool, e = student(student_normalize(images), True)
            at_loss = opts.at_ratio * (at_criterion(b2, t_b2) + at_criterion(b3, t_b3) + at_criterion(b4, t_b4))

        triplet_loss = opts.triplet_ratio * triplet_criterion(e, labels)
        dist_loss = opts.dist_ratio * dist_criterion(e, t_e)
        angle_loss = opts.angle_ratio * angle_criterion(e, t_e)
        quad_loss = opts.quad_ratio * quad_criterion(e, t_e)
        dark_loss = opts.dark_ratio * dark_criterion(e, t_e)

        loss = triplet_loss + dist_loss + angle_loss + quad_loss + dark_loss + at_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        triplet_loss_all.append(triplet_loss.item())
        dist_loss_all.append(dist_loss.item())
        angle_loss_all.append(angle_loss.item())
        quad_loss_all.append(quad_loss.item())
        dark_loss_all.append(dark_loss.item())
        at_loss_all.append(at_loss.item())
        loss_all.append(loss.item())

        train_iter.set_description("[Train][Epoch %d] Triplet: %.5f, Dist: %.5f, Angle: %.5f, Quad: %.5f, Dark: %5f, At: %5f" %
                       (ep, triplet_loss.item(), dist_loss.item(), angle_loss.item(), quad_loss.item(), dark_loss.item(), at_loss.item()))
    loss_mean = torch.Tensor(loss_all).mean().item()
    triplet_mean = torch.Tensor(triplet_loss_all).mean().item()
    dist_mean = torch.Tensor(dist_loss_all).mean().item()
    angle_mean = torch.Tensor(angle_loss_all).mean().item()
    quad_mean = torch.Tensor(quad_loss_all).mean().item()
    dark_mean = torch.Tensor(dark_loss_all).mean().item()
    at_mean = torch.Tensor(at_loss_all).mean().item()

    print('[Epoch %d] Loss: %.5f, Triplet: %.5f, Dist: %.5f, Angle: %.5f, Quad: %.5f, Dark: %.5f At: %.5f\n' %\
          (ep, loss_mean, triplet_mean, dist_mean, angle_mean, quad_mean, dark_mean, at_mean))

    return {
        'loss': loss_mean,
        'triplet': triplet_mean,
        'dist': dist_mean,
        'angle': angle_mean,
        'quad': quad_mean,
        'dark': dark_mean,
        'at': at_mean,
    }


def eval(net, normalize, loader, ep, return_embeddings=False):
    K = opts.recall
    net.eval()
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            output = net(normalize(images))
            embeddings_all.append(output.data)
            labels_all.append(labels.data)
            test_iter.set_description("[Eval][Epoch %d]" % ep)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all, K=K)

        for k, r in zip(K, rec):
            print('[Epoch %d] Recall@%d: [%.4f]\n' % (ep, k, 100 * r))

    if return_embeddings:
        return rec, embeddings_all, labels_all
    return rec


teacher_train_rec = eval(teacher, teacher_normalize, loader_train_eval, 0)
teacher_val_rec = eval(teacher, teacher_normalize, loader_eval, 0)
best_train_rec = eval(student, student_normalize, loader_train_eval, 0)
best_val_rec, best_val_embeddings, best_val_labels = eval(student, student_normalize, loader_eval, 0, return_embeddings=True)

run.log({
    'epoch': 0,
    **_recall_log_dict('teacher/train', teacher_train_rec),
    **_recall_log_dict('teacher/test', teacher_val_rec),
    **_recall_log_dict('eval/train', best_train_rec),
    **_recall_log_dict('eval/test', best_val_rec),
    **_recall_log_dict('best/train', best_train_rec),
    **_recall_log_dict('best/test', best_val_rec),
}, step=0)

_append_history_row(
    epoch=0,
    train_stats=None,
    train_rec=best_train_rec,
    test_rec=best_val_rec,
    best_train_rec=best_train_rec,
    best_test_rec=best_val_rec,
    lr=optimizer.param_groups[0]['lr'],
)
_log_confusion_matrix('eval/test', best_val_embeddings, best_val_labels, step=0)

for epoch in range(1, opts.epochs+1):
    train_stats = train(loader_train_sample, epoch)
    train_recall = eval(student, student_normalize, loader_train_eval, epoch)
    val_recall, val_embeddings, val_labels = eval(student, student_normalize, loader_eval, epoch, return_embeddings=True)

    run.log({
        'epoch': epoch,
        'train/loss': train_stats['loss'],
        'train/triplet_loss': train_stats['triplet'],
        'train/dist_loss': train_stats['dist'],
        'train/angle_loss': train_stats['angle'],
        'train/quad_loss': train_stats['quad'],
        'train/dark_loss': train_stats['dark'],
        'train/at_loss': train_stats['at'],
        **_recall_log_dict('eval/train', train_recall),
        **_recall_log_dict('eval/test', val_recall),
        **_recall_log_dict('best/train', best_train_rec),
        **_recall_log_dict('best/test', best_val_rec),
        'lr': optimizer.param_groups[0]['lr'],
    }, step=epoch)

    for i, _ in enumerate(opts.recall):
        if best_train_rec[i] < train_recall[i]:
            best_train_rec[i] = train_recall[i]

    if best_val_rec[0] < val_recall[0]:
        best_val_rec = list(val_recall)
        best_val_embeddings = val_embeddings
        best_val_labels = val_labels
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))
            best_path = "%s/%s" % (opts.save_dir, "best.pth")
            best_artifact = wandb.Artifact(
                name='distill-model-best-%s' % _name(opts.base).lower(),
                type='model',
                metadata={
                    'epoch': epoch,
                    'best_test_recall_primary': best_val_rec[0],
                    'best_test_recall_k': opts.recall,
                    'best_test_recall_values': best_val_rec,
                },
            )
            best_artifact.add_file(best_path)
            run.log_artifact(best_artifact, aliases=['best'])

    _append_history_row(
        epoch=epoch,
        train_stats=train_stats,
        train_rec=train_recall,
        test_rec=val_recall,
        best_train_rec=best_train_rec,
        best_test_rec=best_val_rec,
        lr=optimizer.param_groups[0]['lr'],
    )

    if opts.save_dir is not None:
        if not os.path.isdir(opts.save_dir):
            os.mkdir(opts.save_dir)
        torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
        last_path = "%s/%s" % (opts.save_dir, "last.pth")
        last_artifact = wandb.Artifact(
            name='distill-model-last-%s' % _name(opts.base).lower(),
            type='model',
            metadata={
                'epoch': epoch,
                'final_test_recall_primary': val_recall[0],
                'final_test_recall_k': opts.recall,
                'final_test_recall_values': val_recall,
            },
        )
        last_artifact.add_file(last_path)
        run.log_artifact(last_artifact, aliases=['last'])
        with open("%s/result.txt" % opts.save_dir, 'w') as f:
            f.write('Recall K: %s\n' % opts.recall)
            f.write('Best Train Recall: %s\n' % [round(v * 100, 4) for v in best_train_rec])
            f.write('Best Test Recall: %s\n' % [round(v * 100, 4) for v in best_val_rec])
            f.write('Final Test Recall: %s\n' % [round(v * 100, 4) for v in val_recall])

    print("Best Train Recall@%d: %.4f" % (primary_recall_k, best_train_rec[0]))
    print("Best Eval Recall@%d: %.4f" % (primary_recall_k, best_val_rec[0]))

_log_confusion_matrix('best/test', best_val_embeddings, best_val_labels, step=opts.epochs)

run.log({'metrics/epoch_table': history_table})

metrics_path = os.path.join(opts.save_dir if opts.save_dir is not None else '.', 'distill_epoch_metrics.csv')
with open(metrics_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(history_columns)
    for row in history_table.data:
        writer.writerow(row)

metrics_artifact = wandb.Artifact(
    name='distill-metrics-%s' % run.id,
    type='metrics',
    metadata={
        'dataset': _name(opts.dataset),
        'base': _name(opts.base),
        'teacher_base': _name(opts.teacher_base),
        'recall': opts.recall,
        'primary_recall_k': primary_recall_k,
    },
)
metrics_artifact.add_file(metrics_path)
run.log_artifact(metrics_artifact)

run.summary['recall_k'] = opts.recall
run.summary['best_train_recall_primary'] = best_train_rec[0]
run.summary['best_test_recall_primary'] = best_val_rec[0]
run.summary['final_test_recall_primary'] = val_recall[0]
for i, k in enumerate(opts.recall):
    run.summary['best_train_recall%d' % k] = best_train_rec[i]
    run.summary['best_test_recall%d' % k] = best_val_rec[i]
    run.summary['final_test_recall%d' % k] = val_recall[i]

run.finish()

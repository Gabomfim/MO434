"""Imprime candidatos de região, role de execução do SageMaker e bucket S3 da
conta autenticada, p/ preencher as flags do launch.py. Somente leitura.

Requer credenciais AWS válidas (ex.: `aws sso login --profile wehandle` e
`export AWS_PROFILE=wehandle`).
"""

import sys


def main():
    try:
        import boto3
    except ImportError:
        print("boto3 não instalado (pip install boto3 sagemaker)."); return 1

    sess = boto3.Session()
    region = sess.region_name or "<não definido: use --region ou AWS_REGION>"
    print(f"região       : {region}")

    try:
        ident = sess.client("sts").get_caller_identity()
        print(f"account      : {ident['Account']}")
        print(f"caller       : {ident['Arn']}")
    except Exception as e:  # noqa: BLE001
        print(f"[!] sem credenciais válidas: {e}")
        return 1

    print("\n-- roles com 'SageMaker' no nome (candidatas a --role) --")
    try:
        iam = sess.client("iam")
        paginator = iam.get_paginator("list_roles")
        found = 0
        for page in paginator.paginate():
            for r in page["Roles"]:
                if "sagemaker" in r["RoleName"].lower():
                    print(f"  {r['Arn']}")
                    found += 1
        if not found:
            print("  (nenhuma; crie uma role de execução do SageMaker)")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] sem permissão p/ listar roles: {e}")

    print("\n-- buckets S3 (candidatos a --bucket) --")
    try:
        for b in sess.client("s3").list_buckets().get("Buckets", []):
            print(f"  {b['Name']}")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] sem permissão p/ listar buckets: {e}")

    try:
        import sagemaker
        print(f"\nsagemaker SDK: {sagemaker.__version__}")
        print(f"default bucket: {sagemaker.Session().default_bucket()}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[i] sagemaker SDK indisponível ou sem default bucket: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Prints candidate region, SageMaker execution role and S3 bucket of the
authenticated account, to fill in the launch.py flags. Read-only.

Requires valid AWS credentials (e.g.: `aws sso login --profile wehandle` and
`export AWS_PROFILE=wehandle`).
"""

import sys


def main():
    try:
        import boto3
    except ImportError:
        print("boto3 not installed (pip install boto3 sagemaker)."); return 1

    sess = boto3.Session()
    region = sess.region_name or "<not set: use --region or AWS_REGION>"
    print(f"region       : {region}")

    try:
        ident = sess.client("sts").get_caller_identity()
        print(f"account      : {ident['Account']}")
        print(f"caller       : {ident['Arn']}")
    except Exception as e:  # noqa: BLE001
        print(f"[!] no valid credentials: {e}")
        return 1

    print("\n-- roles with 'SageMaker' in the name (candidates for --role) --")
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
            print("  (none; create a SageMaker execution role)")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] no permission to list roles: {e}")

    print("\n-- S3 buckets (candidates for --bucket) --")
    try:
        for b in sess.client("s3").list_buckets().get("Buckets", []):
            print(f"  {b['Name']}")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] no permission to list buckets: {e}")

    try:
        import sagemaker
        print(f"\nsagemaker SDK: {sagemaker.__version__}")
        print(f"default bucket: {sagemaker.Session().default_bucket()}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[i] sagemaker SDK unavailable or no default bucket: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

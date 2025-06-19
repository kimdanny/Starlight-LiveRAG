import boto3

AWS_PROFILE_NAME = "<YOUR_PROFILE_NAME>"
AWS_REGION_NAME = "<YOUR_REGION>"


def get_ssm_value(
    key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME
) -> str:
    """Get a cleartext value from AWS SSM."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key)["Parameter"]["Value"]


def get_ssm_secret(
    key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME
):
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]

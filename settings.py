import sys

from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, ValidationError


class Settings(BaseSettings):
    train_size: float = Field(0.6, env='train_size')
    random_state: int = Field(2, env='random_state')
    depth: int = Field(25, env='depth')
    samples: int = Field(2, env='samples')
    path_to_file: str = Field('Social_Network_Ads.csv', env='path_to_file')

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


try:
    load_dotenv()
    settings = Settings(_env_file=".env")
    logger.info('settings loaded successfully')

except ValidationError as ex:
    ex_errors = ex.errors()
    print('Error with .env file!')
    for error in ex_errors:
        print(f'    check "{error["loc"][0]}"')

    sys.exit(1)


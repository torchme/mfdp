"""Backend."""
from __future__ import annotations

import sys

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from .baseline import popular_items
from .utils import read_data

app = FastAPI(
    title='Popular Items',
    description='Recommend popular items for a given genre',
    version='1.0.0',
)


class User(BaseModel):
    """User model."""

    username: int
    email: str


def get_user(username: int, data_user):
    """Get user."""
    # hotmake change to BD
    output = data_user[data_user['user_id'] == username].values.squeeze()

    if output.shape[0] != 3:
        output = output[:3]

    if output[2] == 0.0:
        output[2] = 'male'
    elif output[2] == 1.0:
        output[2] = 'female'
    else:
        output[2] = 'Unknown'
    return list(output)


@app.get('/user/{user_id}')
def home(user_id: int):
    """Homepage app.

    Returns
    -------
    _type_
        _description_
    """
    path = sys.path[0]
    data, data_user, data_items = read_data(path)

    user_info = get_user(username=user_id, data_user=data_user)
    items = list(popular_items(data=data, data_items=data_items))

    return {
        'User': {
            'user_id': user_info[0],
            'age': user_info[1],
            'sex': user_info[2],
        },
        'Popular Items': items,
        'Personal Recomendations': 'GET',
    }


if __name__ == '__main__':
    logger.warning('Running in development mode.')

    uvicorn.run(app, host='localhost', port=8001, log_level='debug')

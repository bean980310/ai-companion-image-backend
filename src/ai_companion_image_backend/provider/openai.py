import os
import base64
import traceback
from typing import Any

from PIL import Image, ImageFile

import openai
from openai import OpenAI
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent
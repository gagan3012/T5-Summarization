import argparse
import os
import pprint
import subprocess
import sys

import yaml

arg_parser = argparse.ArgumentParser(
    description="T5 Summarisation Using Pytorch Lightning", prog="t5s"
)
# Command choice
command_subparser = arg_parser.add_subparsers(
    dest="command", help="command (refer commands section in documentation)"

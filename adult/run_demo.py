# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstration script using the MNIST dataset."""

import os
import tempfile
from typing import Literal, Optional

import multiprocessing

import fire  # type: ignore

from declearn.test_utils import generate_ssl_certificates, make_importable
from declearn.utils import run_as_processes

# Perform local imports.
# pylint: disable=wrong-import-position, wrong-import-order
with make_importable(os.path.dirname(__file__)):
    from run_client import run_client
    from run_server import run_server
# pylint: enable=wrong-import-position, wrong-import-order


def run_demo(
    nb_clients: int = 5,
    scheme: Literal["iid", "labels", "biased"] = "iid",
    seed: Optional[int] = None,
) -> None:
    """Run a server and its clients using multiprocessing.

    Parameters
    ------
    n_clients: int
        number of clients to run.
    data_folder: str
        Relative path to the folder holding client's data
    """
    # Use the adult data for this demo.
    data_folder = "../adult-data"
    # Use a temporary directory for single-use self-signed SSL files.
    with tempfile.TemporaryDirectory() as folder:
        print("Generating self-signed SSL certificates...")
        # Generate self-signed SSL certificates and gather their paths.
        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(folder)
        # Specify the server and client routines that need executing.
        server = (run_server, (nb_clients, sv_cert, sv_pkey))
        client_kwargs = {
            "data_folder": data_folder,
            "ca_cert": ca_cert,
            "verbose": False,
        }
        clients = [
            (run_client, (f"client_{idx}",), client_kwargs) for idx in range(nb_clients)
        ]
        # Run routines in isolated processes. Raise if any failed.
        print("Running server and clients...")
        success, outp = run_as_processes(server, *clients)
        if not success:
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                "\n".join(str(e) for e in outp if isinstance(e, RuntimeError))
            )


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    fire.Fire(run_demo)

#!/usr/bin/env python3

import os
import os.path as osp
import sys

PATH = osp.join(osp.abspath(osp.dirname(__file__)), "code")
if PATH not in sys.path:
    sys.path.append(str(PATH))

from gnnet24.main import main

if __name__ == "__main__":
    main()

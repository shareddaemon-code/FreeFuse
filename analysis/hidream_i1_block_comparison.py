#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HiDream i1 block comparison launcher.

HiDream i1 currently follows the same single-stream block sweep protocol as
Z-Image. This launcher exists to provide a first-class architecture entrypoint.
"""

from analysis.zit_block_comparison import main


if __name__ == "__main__":
    main()

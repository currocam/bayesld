#!/usr/bin/env -S uv run --script --isolated
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bayesld @ git+https://github.com/currocam/bayesld.git@c171824",
# ]
# ///

import bayesld

print("Successfully imported bayesld.")
print("Output of bayesld.linear_bins():")
print(bayesld.linear_bins())

"""Static reference data that production code imports.

Deliberately NOT under data/. That directory is gitignored (.gitignore:41) and
excluded from the Space deploy, so a module living there is invisible to CI —
it never enters the checkout and therefore never reaches production. That is
how `from data.stocks_list import ...` came to raise ModuleNotFoundError on the
Space, breaking the index_data scheduler job and GET /api/heatmap/sectors while
working perfectly on every developer's machine.

data/ is for data. Code that production imports belongs in a tracked package.
"""

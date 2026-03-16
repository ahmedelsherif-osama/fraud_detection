import os

for root, dirs, files in os.walk("."):
    for d in dirs:
        init_file = os.path.join(root, d, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, "a").close()

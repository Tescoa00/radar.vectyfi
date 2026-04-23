from setuptools import find_packages, setup

with open("requirements-api.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='vectyfi',
      version="0.0.12",
      description="Vectyfi Model (api_pred)",
      license="MIT",
      author="Vectifi Team @ Le Wagon",
      author_email="contact@lewagon.org",
      #url="https://github.com/lewagon/taxi-fare",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)

# project_name 
project_description

## Installation

Clone the repository with:

```bash
git clone project_url 
```

then to install it in normal mode:

```bash
pip install .
```

Use poetry to install the latest version in developer mode, remember to also
install the pre-commits!

```bash
poetry install --with docs,analysis
pre-commit install
```

## License

project_name is licensed under the [Apache License 2.0](LICENSE). See the [LICENSE](LICENSE) file for details.

## Using the template

By default, crating a new project from this template will trigger a one-time github action (credits largely to [rochacbruno](https://github.com/rochacbruno/python-project-template.git)) that checks if the [.github/template.yml](.github/template.yml) is present. If so, it will automatically rename the project, delete the file and this part of the README. You can then delete the [rename_project.yml](.github/workflows/rename_project.yml) and [rename_project.sh](.github/rename_project.sh) files and uncomment the third line of [analysis.yml](.github/workflows/analysis.yml) to activate testing on push. Your project is now ready!

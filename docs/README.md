# mytender.io Documentation

This folder contains the Sphinx documentation for the mytender.io system.

## Structure

- `source/`: Contains the source files for the documentation
  - `_static/`: Contains static files like images
  - `conf.py`: Sphinx configuration file
  - `*.rst`: ReStructuredText files for documentation
- `rag_architecture_diagram.py`: Python script to generate the RAG architecture diagram
- `Makefile`: Makefile for building the documentation
- `build_docs.sh`: Shell script to automate the documentation build process
- `serve_docs.sh`: Shell script to serve the documentation via HTTP
- `deploy_docs.sh`: Shell script to manually deploy documentation to the server
- `nginx-sphinx.conf`: Nginx configuration for serving the documentation
- `nginx-sphinx-dev.conf`: Nginx configuration for serving the dev documentation

## Building the Documentation

### Using the Automated Scripts (Recommended)

Simply run the build script from the project root:

```bash
./build_docs.sh
```

This will:
1. Create a virtual environment if it doesn't exist
2. Install all required dependencies
3. Generate the RAG architecture diagram
4. Build the HTML documentation

### Viewing the Documentation

To view the documentation in a web browser, run:

```bash
./serve_docs.sh
```

This will start a local web server and make the documentation available at http://localhost:8000/

### Online Documentation

The documentation is automatically built and deployed when changes are pushed to the main or dev branches:

- Documentation URL: https://docs.mytender.io/ (or your configured domain)

## CI/CD Pipeline

The documentation is automatically built and deployed using GitHub Actions:

1. When changes are pushed to the `docs/` directory on the main or dev branch
2. The workflow defined in `.github/workflows/publish_sphinx.yml` is triggered
3. The documentation is built using the same build script as local development
4. Any existing files in the target directory are removed to ensure a clean deployment
5. The built HTML is copied to the server in the target directory `/home/ec2-user/sphinx_doc_html/`

## Manual Deployment

To manually deploy the documentation to the server, you can use the provided script:

```bash
./deploy_docs.sh
```

The script will:
1. Build the documentation if it doesn't exist
2. Clean the target directory on the server
3. Copy the new documentation files

## Manual Build Process

If you prefer to build the documentation manually:

1. Generate the RAG architecture diagram:

```bash
cd docs
python rag_architecture_diagram.py
```

2. Build the HTML documentation:

```bash
cd docs
make html
```

3. View the documentation:

```bash
open build/html/index.html
```

## Requirements

To build the documentation, you need the following Python packages:

- sphinx>=4.0.0
- sphinx_rtd_theme>=1.0.0
- matplotlib>=3.5.0
- numpy>=1.20.0

These are specified in `requirements.txt` and will be installed automatically by the build script.

## Updating the Documentation

1. Edit the RST files in the `source/` directory
2. Run `./build_docs.sh` to rebuild the documentation
3. View the documentation using `./serve_docs.sh`
4. Push your changes to either the main or dev branch to trigger automatic deployment 
# Stable Diffusion Virtual Try-On API

This repository provides the implementation of a Stable Diffusion inference model to offer a virtual try-on as a service. It leverages the serverless framework "Modal" to serve an API that allows users to upload images and try on virtual clothes.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to create an API service that uses Stable Diffusion for virtual try-on applications. Users can upload images of themselves and garments to see how they would look wearing the clothes.

## Features

- **Stable Diffusion Model**: Utilizes Stable Diffusion for high-quality virtual try-on images.
- **Serverless Framework**: Implements the Modal serverless framework for efficient and scalable API deployment.
- **Easy Integration**: Simple API endpoints for seamless integration into other applications.

## Requirements

- Python 3.8+
- Docker
- Modal account and CLI

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/josemiguel/viton.git
    cd viton
    ```

2. **Install dependencies**:
    ```bash
    pip install modal
    ```

3. **Set up Modal CLI**:
    ```bash
    python3 -m modal setup 
    ```

4. **Deploy the service**:
    ```bash
    modal deploy viton_api.py
    ```

## Usage

After deploying the service, you can use the provided API endpoints to perform virtual try-on operations.

### Example Request

```bash
curl -s "https://your-modal-service-url/api/v1/tryon" \
    -H "Content-Type: application/json" \
    -d '{
        "human_img_url": "https://example.com/user.jpg",
        "garm_img_url": "https://example.com/clothing.jpg",
        "prompt": "t-shirt",
        "user_id": "1234"
    }'


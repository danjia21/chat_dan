#!/bin/bash

check_git_lfs() {
    if ! command -v git-lfs &> /dev/null; then
        echo "Git LFS is not installed. Please install git-lfs to continue."
        exit 1
    fi
}

download_weights() {
    if [ -d "hf_models/flan-t5-large" ]; then
        echo "flan-t5-large directory exists, skipping clone."
    else
        git clone https://huggingface.co/google/flan-t5-large/tree/main hf_models/flan-t5-large
    fi
}

# Main function
main() {
    check_git_lfs
    download_weights
}

main

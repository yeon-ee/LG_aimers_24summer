# LG Aimers Anomaly Detection

## Get Started

### Environment Setup

To ensure compatibility, please set up a Python virtual environment with Python version 3.9.

### Step 1: Set Up the Python Environment

1. **Install Miniforge** (a minimal installer for Conda specific to your system architecture):
    
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    export PATH="$HOME/miniforge3/bin:$PATH"
    source ~/.bashrc
    ```
    
2. **Install Mamba** (a fast version of Conda):
    
    ```bash
    conda install mamba -n base -c conda-forge
    
    ```
    
3. **Create a Python 3.9 Virtual Environment**:
    
    ```bash
    mamba create -n lg python==3.9
    mamba activate lg
    
    ```
    

### Step 2: Clone the Repository

To avoid conflicts when pushing your changes, make sure to clone the forked repository rather than the original repository.

1. **Generate an SSH Key**:
    
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    
    ```
    
    - Press `Enter` to accept the default file location.
    - If prompted, set a passphrase (or press `Enter` to skip).
2. **Add the SSH Key to GitHub**:
    - Copy the SSH public key to your clipboard:
        
        ```bash
        cat ~/.ssh/id_ed25519.pub
        
        ```
        
    - Go to GitHub > Settings > SSH and GPG keys > New SSH key.
    - Give the key a title, paste the copied key, and save it.
3. **Verify the SSH Connection**:
    
    ```bash
    ssh -T git@github.com
    
    ```
    
4. **Clone Your Forked Repository**:
    
    ```bash
    git clone git@github.com:jkyoon2/LG_aimers_24summer.git
    cd LG_aimers_24summer
    
    ```

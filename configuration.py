import os


def create_config():
    # Define the directory and file paths
    directory = ".streamlit"
    file_path = os.path.join(directory, "config.toml")

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    # Define the content for the config.toml file with custom theme settings
    config_content = """
    [theme]
    primaryColor="blue"
    backgroundColor="black"
    secondaryBackgroundColor="#2F2F2F"
    textColor="white"
    font="serif"
    """

    # Create the config.toml file and write content to it
    with open(file_path, "w") as file:
        file.write(config_content)
        print(f"File '{file_path}' created and content written.")

    print("Setup completed.")

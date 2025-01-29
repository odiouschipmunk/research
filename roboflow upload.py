from roboflow import Roboflow
import random, os

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="NiryUpt2WRXVD0j2DPPL")

# Retrieve your current workspace and project name
print(rf.workspace())

# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
workspaceId = "squash-vision"
projectId = "white-squash-ball"
project = rf.workspace(workspaceId).project(projectId)


def get_random_file(directory):
    # Get list of files
    files = os.listdir(directory)
    # Return random file
    return random.choice(files)


# Upload the image to your project
for i in range(5000):
    # find a random file in the directory and upload it
    project.upload("output frames fvel/" + get_random_file("output frames fvel/"))
    print(f"Uploaded image {i + 1}")

"""
Optional Parameters:
- num_retry_uploads: Number of retries for uploading the image in case of failure.
- batch_name: Upload the image to a specific batch.
- split: Upload the image to a specific split.
- tag: Store metadata as a tag on the image.
- sequence_number: [Optional] If you want to keep the order of your images in the dataset, pass sequence_number and sequence_size..
- sequence_size: [Optional] The total number of images in the sequence. Defaults to 100,000 if not set.
"""

project.upload(
    image_path="UPLOAD_IMAGE.jpg",
    batch_name="YOUR_BATCH_NAME",
    split="train",
    num_retry_uploads=3,
    tag="YOUR_TAG_NAME",
    sequence_number=99,
    sequence_size=100,
)

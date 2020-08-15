# pytorch-style-transfer
## Intro
This is a simple style transfer in pytorch. You use a content image and style image, and set image size and step count. Finally you get a output image that is both similar to content image and style image.
## How to use
```python
root_dir = "./data/"
transfer = StyleTransfer(
            size = (320, 500), 
            content_path = root_dir + "content.jpg",
            style_path = root_dir + "style.jpg",
            step_count=100,
            save_path= root_dir + "output.jpg"
           )
transfer.load_image()
transfer.run()
```

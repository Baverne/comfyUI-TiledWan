class TiledWanPassThrough:
    """
    A simple pass-through node that outputs the same image it receives as input.
    This is a dummy node for the TiledWan node set.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for this node.
        """
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    OUTPUT_NODE = False
    CATEGORY = "TiledWan"

    def process(self, image):
        """
        Simply return the input image unchanged.
        
        Args:
            image: Input image tensor
            
        Returns:
            tuple: The same image tensor
        """
        # This is a pass-through node - just return the input image
        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanPassThrough": TiledWanPassThrough
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanPassThrough": "TiledWan Pass Through"
}

class Layer:
    def forward(self, input):
        """
        Perform the forward pass of the layer.
        
        Parameters:
        input: The input data to the layer.
        
        Returns:
        The output of the layer.
        """
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, output_gradient):
        """
        Perform the backward pass of the layer.
        
        Parameters:
        output_gradient: The gradient of the loss with respect to the output of this layer.
        
        Returns:
        The gradient of the loss with respect to the input of this layer.
        """
        raise NotImplementedError("Backward pass not implemented.")
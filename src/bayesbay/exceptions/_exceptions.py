class DimensionalityException(Exception):
    """
    Exception raised when trying to add/remove a dimension in a
    parameter space that has already reached the maximum/minimum number of
    allowed dimensions (see :class:`bayesbay.parameterization.ParameterSpace`).
    """

    def __init__(self, move):
        message = (
            "Error occured when trying to %s a Voronoi site."
            " The %s number of Voronoi cells had already been reached."
        )
        words = ("add", "maximum") if move == "Birth" else ("remove", "minimum")
        self.message = message % words
        super().__init__(self.message)

    def __str__(self):
        return self.message


class ForwardException(Exception):
    """
    Exception raised when a user-provided forward function raises an error
    """
    def __init__(self, original_exc):
        self.message = "error occurred when running the forward function - " + (
            original_exc.message
            if hasattr(original_exc, "message")
            else f"{type(original_exc).__name__}: {str(original_exc)}"
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InitException(Exception):
    """
    Exception raised when users try to access a certain field that hasn't been
    intialized yet
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UserFunctionException(Exception):
    """Exception raised when a user-provided function raises an exception"""

    def __init__(self, original_exc):
        self.message = "error occurred when running the user-provided function - " + (
            original_exc.message
            if hasattr(original_exc, "message")
            else f"{type(original_exc).__name__}: {str(original_exc)}"
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message

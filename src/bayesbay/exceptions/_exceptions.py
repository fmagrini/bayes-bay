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


class InvalidProposalException(Exception):
    """Exception indicating that a proposed model has zero posterior probability.

    Users may raise this exception from a perturbation, forward function, or
    custom prior when the proposed state is well formed but deliberately deemed
    inadmissible. Markov chains treat it as a normal rejected proposal, including
    the corresponding self-transition in the chain history.
    """


class OutOfDomainException(InvalidProposalException):
    """Exception raised when a position-dependent prior is evaluated at a position
    outside of the specified domain

    This is a subclass of ``InvalidProposalException``: the Markov chains treat
    an out-of-domain evaluation as a rejected proposal (counted in the chain
    statistics under ``exceptions``) rather than interrupting the sampling.
    """

    def __init__(self, variable_name, x):
        self.variable_name = variable_name
        self.x = x
        self.message = (
            f"the position-dependent prior '{variable_name}' was evaluated at "
            f"a position ({x}) outside of the domain specified earlier"
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message

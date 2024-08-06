import torch
import torch.nn as nn

class RepresentationLayer(nn.Module):
    
    """
    Class implementing a representation layer accumulating gradients.
    """


    ######################## PUBLIC ATTRIBUTE #########################


    # Set the available distributions to sample the representations
    # from.
    AVAILABLE_DISTS = ["normal"]


    ######################### INITIALIZATION ##########################

    
    def __init__(self,
                 values = None,
                 dist = "normal",
                 dist_options = None):
        """Initialize a representation layer.

        Parameters
        ----------
        values : ``torch.Tensor``, optional
            A tensor used to initialize the representations in
            the layer.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations in the tensor.

            * The second dimension has a length equal to the
              dimensionality of the representations.

            If the tensor is not passed, the representations will be
            initialized by sampling the distribution specified
            by ``dist``.

        dist : ``str``, {``"normal"``}, default: ``"normal"``
            The name of the distribution used to sample the
            representations, if no ``values`` are passed.

            By default, the distribution is a ``"normal"``
            distribution.

        dist_options : ``dict``, optional
            A dictionary containing the parameters to sample the
            representations from the distribution, if no ``values``
            are passed.

            For any distribution the following keys and associated
            parameters must be provided:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            If ``dist`` is ``"normal"``, the dictionary must contain
            these additional key/value pairs:

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.
        """
        
        # Initialize an instance of the 'nn.Module' class.
        super().__init__()
        
        # Initialize the gradients with respect to the representations
        # None.
        self.dz = None

        # If a tensor of values was passed
        if values is not None:

            # Set the options used to initialize the representations
            # to an empty dictionary, since they have not been 
            # sampled from any distribution.
            self._options = {}

            # Get the number of representations, the
            # dimensionality of the representations, and the values
            # of the representations from the tensor.
            self._n_rep, self._dim, self._z = \
                self._get_rep_from_values(values = values)      
        
        # Otherwise
        else:

            # If the representations are to be sampled from a normal
            # distribution
            if dist == "normal":

                # Sample the representations from a normal
                # distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_normal(options = dist_options)

            # Otherwise
            else:

                # Raise an error.
                available_dists_str = \
                    ", ".join(f'{d}' for d in self.AVAILABLE_DISTS)
                errstr = \
                    f"Unsupported distribution '{dist}'. The only " \
                    "distributions from which it is possible to " \
                    "sample the representations are: " \
                    f"{available_dists_str}."
                raise ValueError(errstr)


    def _get_rep_from_values(self,
                             values):
        """Get the representations from a given tensor of values.

        Parameters
        ----------
        values : ``torch.Tensor``
            The tensor used to initialize the representations.

        Returns
        -------
        n_rep : ``int``
            The number of representations found in the input tensor.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """

        # Get the number of representations from the first dimension of
        # the tensor.
        n_rep = values.shape[0]
        
        # Get the dimensionality of the representations from the last
        # dimension of the tensor.
        dim = values.shape[-1]

        # Initialize a tensor with the representations.
        z = nn.Parameter(torch.zeros_like(values), 
                         requires_grad = True)

        # Fill the tensor with the given values.
        with torch.no_grad():
            z.copy_(values)

        # Return the number of representations, the dimensionality of
        # the representations, and the values of the representations.
        return n_rep, \
               dim, \
               z


    def _get_rep_from_normal(self,
                             options):
        """Get the representations by sampling from a normal
        distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a normal distribution.

            The dictionary must contains the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the normal distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the normal distribution.

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.

        Returns
        -------
        n_rep : ``int``
            The number of representations found in the input tensor.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the mean of the normal distribution from which the
        # representations should be samples.
        mean = options["mean"]

        # Get the standard deviation of the normal distribution
        # from which the representations should be sampled.
        stddev = options["stddev"]

        # Get the values of the representations.
        z = \
            nn.Parameter(\
                torch.normal(mean,
                             stddev,
                             size = (n_rep, dim),
                             requires_grad = True))
        
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "normal",
                "mean" : mean,
                "stddev" : stddev}


    ########################### PROPERTIES ############################


    @property
    def n_rep(self):
        """The number of representations in the layer.
        """

        return self._n_rep


    @n_rep.setter
    def n_rep(self,
              value):
        """Raise an exception if the user tries to modify the value
        of ``n_rep`` after initialization.
        """
        
        errstr = \
            "The value of 'n_samples' is set at initialization and " \
            "cannot be changed. If you want to change the number " \
            "of representations in the layer, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def dim(self):
        """The dimensionality of the representations.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify the value of
        ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' is set at initialization and cannot " \
            "be changed. If you want to change the dimensionality " \
            "of the representations stored in the layer, initialize " \
            f"a new instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def options(self):
        """The dictionary ot options used to generate the
        representations, if no values were passed when initializing
        the layer.
        """

        return self._options


    @options.setter
    def options(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``options`` after initialization.
        """
        
        errstr = \
            "The value of 'options' is set at initialization and " \
            "cannot be changed. If you want to change the options " \
            "used to generate the representations, initialize a " \
            f"new instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)
    

    @property
    def z(self):
        """The values of the representations.
        """

        return self._z


    @z.setter
    def z(self,
          value):
        """Raise an exception if the user tries to modify the value of
        ``z`` after initialization.
        """
        
        errstr = \
            "The value of 'z' is set at initialization and cannot " \
            "be changed. If you want to change the values of the " \
            "representations stored in the layer, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                ixs = None):
        """Forward pass - it returns the values of the representations.

        You can select a subset of representations to be returned using
        their numerical indexes.

        Parameters
        ----------
        ixs : ``list``, optional
            The indexes of the samples whose representations should
            be returned. If not passed, all representations will be
            returned.

        Returns
        -------
        reps : ``torch.Tensor``
            A tensor containing the values of the representations for
            the samples of interest.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """

        # If no indexes were provided
        if ixs is None:
            
            # Return the values for all representations.
            return self.z
        
        # Otherwise
        else:

            # Return the values for the representations of the
            # samples corresponding to the given indexes.
            return self.z[ixs]


    def rescale(self):
        """Rescale the representations by subtracting the mean of
        the representations' values from each of them and dividing
        each of them by the standard deviation of all representations.

        Given :math:`N` samples, we can indicate with :math:`z^{n}`
        the value of the representation of sample :math:`x^{n}`.

        Therefore, the rescaled value of the representation
        :math:`z^{n}_{rescaled}` will be:
        
        .. math::

           z^{n}_{rescaled} = \\frac{z^{n} - \\bar{z}}{s}

        Where :math:`\\bar{z}` is the mean of the representations'
        values and :math:`s` is the standard deviation.

        Returns
        -------
        reps_rescaled : ``torch.Tensor``
            The rescaled values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """
        
        # Flatten the tensor containing the representations' values.
        z_flat = torch.flatten(self.z.cpu().detach())
        
        # Get the mean and the standard deviation of the
        # representations.
        sd, m = torch.std_mean(z_flat)
        
        # Disable the calculation of the gradients.
        with torch.no_grad():

            # Subtract the mean value of all representations' values
            # from each of the representation's value.
            self.z -= m

            # Divide each representation's value by the standard
            # deviation of all representations' values.
            self.z /= sd
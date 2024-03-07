Code contribution
=================

Steps for submitting your code
------------------------------

When contributing code follow this checklist:

    #. Fork the repository on GitHub.
    #. Create an issue with the desired feature or bug fix.
    #. Make your modifications or additions in a feature branch.
    #. Make changes and commit your changes using a descriptive commit message.
    #. Provide tests for your changes, and ensure they all pass.
    #. Provide documentation for your changes, in accordance with the style of the rest of the project (see :ref:`style_guide`).
    #. Create a pull request to ROSE main branch. The ROSE team will review and discuss your Pull Request with you.

For any questions, please get in contact with one of the members of :doc:`authors`.


.. _style_guide:

Code style guide
----------------
The additional features should follow the style of the ROSE project.

The class or function name should be clear and descriptive of te functionality it provides.

There should be a docstring at the beginning of the class or function describing its purpose and usage.
The docstring should be in the form of a triple-quoted string.

Please, avoid inheritance, and favour composition when writing your code.

An example of a class:

.. code-block::

    class Wheel(ElementModelPart):
        """
        Wheel model part class.
        This class bases from :class:`rose.model.model_part.ElementModelPart`.

        :Attributes:

            - :self.mass: wheel mass
            - :self.total_static_load: total static load of the wheel
            - :self.distances:  Distance from the zero coordinate to the wheel at every time step
            - :self.active_n_dof: Number of active degrees of freedom of the wheel
        """
        def __init__(self):
            super().__init__()
            self.mass = 0
            self.total_static_load = None
            self.distances = None
            self.active_n_dof = None
            self.dofs = None


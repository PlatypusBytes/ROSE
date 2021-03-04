.. _extend:

Contributing
============


Structure
---------



Testing
-------




Documenting
-----------

We use Sphinx and thus reStructuredText `.rst` files to write our documentation. It is good to read the basics about
reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html.

**Always** write docstrings for your classes, methods or functions. These docstrings are used
automatically in the documentation. Many IDEs support generating a docstring on the fly.
We use the @@ style of docstrings:

.. code-block:: python

    def func(arg1: int, arg2: str) -> bool:
        """Summary line.

        Extended description of function.

        Args:
            arg1: Description of arg1
            arg2: Description of arg2

        Returns:
            Description of return value

        """
        return True

You can find the Sphinx documentation about this style here: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html


Inheritance
-----------

Type hinting
------------


Adding requirements
-------------------


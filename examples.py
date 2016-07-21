# -*- coding: utf-8 -*-
"""
Examples of NeuralCode

AUTHORS:

- Ethan Petersen (2015-09) [initial version]

This file constructs some examples of NeuralCodes.

The examples are accessible by typing: ``neuralcodes.example()``
"""

class NeuralCodeExamples():
    r"""
    Some examples of neuralcodes.

    Here are the available examples; you can also type
    ``neuralcodes.``  and hit tab to get a list:

    - :meth:`Canonical`
    - :meth:`Factored_Canonical`
    - :meth:`Groebner`
    - :meth:`RF_Structure`

    EXAMPLES::

        sage: nc = neuralcodes.Canonical()
        Ideal (x1*x2, x0*x1 + x0, x1*x2 + x1 + x2 + 1, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2

        sage: f = neuralcodes.Factored_Canonical()
        [x2 * x1, (x1 + 1) * x0, (x2 + 1) * (x1 + 1), x2 * x0]

        sage: g = neuralcodes.Groebner()
        Ideal (x0*x2, x1 + x2 + 1) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2

        sage: rf = neuralcodes.RF_Structure()
        Intersection of U_['2', '1'] is empty
    	Intersection of U_['0'] is a subset of Union of U_['1']
    	X = Union of U_['2', '1']
    	Intersection of U_['2', '0'] is empty
    """

    def __call__(self):
        r"""
        If neuralcodes() is executed, return a helpful message.

        INPUT:

        None

        OUTPUT:

        None

        EXAMPLES::

            sage: neuralcodes()
            Try neuralcodes.FOO() where FOO is in the list:

                Canonical, Factored_Canonical, Groebner, RF_Structure
        """
        print 'Try neuralcodes.FOO() where FOO is in the list:\n'
        print "    " + ", ".join([str(i) for i in dir(neuralcodes) if i[0]!='_'])

    def Canonical(self):
        """
        The canonical form of the ideal corresponding to ['001','010','110'].

        INPUT:

        None

        OUTPUT:

        - Ideal

        EXAMPLES::

            sage: s = neuralcodes.Canonical()
            Ideal (x1*x2, x0*x1 + x0, x1*x2 + x1 + x2 + 1, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2
        """
        nc = NeuralCode(['001','010','110'])
        return nc.get_canonical()

    def Factored_Canonical(self):
        """
        The factored canonical form of the ideal corresponding to ['001','010','110'].

        INPUT:

        None

        OUTPUT:

        - List

        EXAMPLES::

            sage: s = neuralcodes.factored_canonical()
            [x2 * x1, (x1 + 1) * x0, (x2 + 1) * (x1 + 1), x2 * x0]
        """
        nc = NeuralCode(['001','010','110'])
        return nc.factored_canonical()

    def Groebner(self):
        """
        The groebner basis of the ideal corresponding to ['001','010','110'].

        INPUT:

        None

        OUTPUT:

        - Ideal

        EXAMPLES::

            sage: s = neuralcodes.get_groebner_basis()
            [x2 * x1, (x1 + 1) * x0, (x2 + 1) * (x1 + 1), x2 * x0]
        """
        nc = NeuralCode(['001','010','110'])
        return nc.get_groebner_basis()

    def RF_Structure(self):
        """
        The RF structure corresponding to ['001','010','110'].

        INPUT:

        None

        OUTPUT:

        None

        EXAMPLES::

            sage: s = neuralcodes.get_canonical_RF_structure()
            Intersection of U_['2', '1'] is empty
    		Intersection of U_['0'] is a subset of Union of U_['1']
    		X = Union of U_['2', '1']
    		Intersection of U_['2', '0'] is empty
        """
        nc = NeuralCode(['001','010','110'])
        return nc.get_canonical_RF_structure()
        
neuralcodes = NeuralCodeExamples()
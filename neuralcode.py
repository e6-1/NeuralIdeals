import itertools
import time
import math
 
from multiprocessing.pool import ThreadPool
from itertools import tee, izip
from sage.rings.polynomial import *
from sage.rings.polynomial.pbori import *
from sage.rings.ideal import *
r"""
Collection of algorithms pertaining to the neural ring.

The following is based around implementations of two algorithms described in the paper, "The Neural Ring: An Algebraic Tool for Analyzing the Intrinsic Structure of Neural Codes" by Curto, C., Itskov, V., Veliz-Cuba, A., and Youngs, N. The first is a primary decomposition of pseudo-monomials, which is used in the second algorithm that computes a canonical form. Along with these are several other methods that are related to neural codes or neural ideals. Additionally, the iterative algorithm to compute the canonical form outlined in "Neural ring homomorphisms and maps between neural codes" by Dr. Carina Curto and Dr. Nora Youngs has also been implemented. 

AUTHORS:

- Ethan Petersen (2015-09-13): initial version

- Dane Miyata (2015-09-13): initial version

- Ryan Kruse (2015-09-13): initial version

- Ihmar Aldana (2015-09-13): initial version

EXAMPLES::

The software's main purpose is to compute canonical forms of neural ideals, and we'll begin with this example::
    
    sage: code = NeuralCode(['001','010','110'])
    sage: code.canonical()
    Ideal (x1*x2, x0*x1 + x0, x1*x2 + x1 + x2 + 1, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2
    
To read RF structures, it's easiest when the generators of the canonical form are factored::
    sage: code.factored_canonical()
    [x2 * x1, (x1 + 1) * x0, (x2 + 1) * (x1 + 1), x2 * x0]
    
Or we can simply retrive the RF structure::
    sage: code.canonical_RF_structure()
    Intersection of U_['2', '1'] is empty
    Intersection of U_['0'] is a subset of Union of U_['1']
    X = Union of U_['2', '1']
    Intersection of U_['2', '0'] is empty
    
We can investigate the groebner basis of the neural code::
    sage: code.groebner_basis()
    Ideal (x0*x2, x1 + x2 + 1) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2

Other methods include determining if a neural code is a simplicial code::
    sage: is_simplicial(code.Codes)
    False
    
    sage: is_simplicial(['000','001','010','100','110','011','101','111'])
    True
    
Computing the groebner fan in the boolean ring::
    sage: code.groebner_fan()
    [Ideal (x1 + x2 + 1, x0*x1 + x0) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2, Ideal (x1 + x2 + 1, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite     Field of size 2]
    
Computing the universal groebner basis in the boolean ring::
    sage: code.universal_groebner_basis()
    Ideal (x1 + x2 + 1, x0*x1 + x0, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2
    
In constructing a neural code object, there are two arguments: the neural code, and an optional argument that is the term order. This term order will be used in the ring where
all member methods do computation. For example, if we change the order to 'degrevlex', groebner_basis() will compute the groebner basis with that term order. 

Additionally, the canonical() method also takes optional arguments. The first will determine which algorithm will be used to compute the canonical form. Either the iterative algorithm outlined by Dr. Carina Curto and Dr. Nora Young in "Neural ring homomorphisms and maps between neural codes" will be used or their algorithm in "The Neural Ring" will be chosen. The second will determine which algorithm to use for a primary decomposition step: 'pm' will use the pseudo-monomial algorithm, 'sy' and 'gtz' will  use the shimoyama-yokoyama algorithm or gianni-trager-zacharias algorithm, respectively. We also noticed that the canonical() runtime begins to rise significantly with higher dimension and more code words. To partially address this, we have parallelized portions of the algorithm used to get the canonical form. To take advantage of this, we use optional arguments::
    sage: code.canonical('pm', True, 3)
    Ideal (x1*x2, x0*x1 + x0, x1*x2 + x1 + x2 + 1, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2
    
The second argument is a boolean: True if we want to use parallelized portions, False if we don't. The third argument is for the number of parallel processes.
There are several other functions specifically designed for research in the neural ring. First, compare_all_groebner_canonical(d) will print a comparison of all of the groebner bases and canonical forms for every possible set of codewords in "d" dimension, except for when the canonical form
is the zero ideal::
    sage: compare_all_groebner_canonical(2)
    Differences: 2 Equal bases: 12

    Different Bases:

    Codes                                             |Groebner                                                                   |Canonical
    ['00', '11']                                      |[x0 + x1]                                                                  |[(x1 + 1) * x0, x1 * (x0 + 1)]
    ['01', '10']                                      |[x0 + x1 + 1]                                                              |[(x1 + 1) * (x0 + 1), x1 * x0]

    Equal Bases:

    Codes                                             |Groebner                                                                        |Canonical
    ['00']                                            |[x1, x0]                                                                        |[x0, x1]
    ['01']                                            |[x0, x1 + 1]                                                                    |[x0, x1 + 1]
    ['10']                                            |[x1, x0 + 1]                                                                    |[x0 + 1, x1]
    ['11']                                            |[x1 + 1, x0 + 1]                                                                |[x0 + 1, x1 + 1]
    ['00', '01']                                      |[x0]                                                                            |[x0]
    ['00', '10']                                      |[x1]                                                                            |[x1]
    ['01', '11']                                      |[x1 + 1]                                                                        |[x1 + 1]
    ['10', '11']                                      |[x0 + 1]                                                                        |[x0 + 1]
    ['00', '01', '10']                                |[x1 * x0]                                                                       |[x1 * x0]
    ['00', '01', '11']                                |[(x1 + 1) * x0]                                                                 |[(x1 + 1) * x0]
    ['00', '10', '11']                                |[x1 * (x0 + 1)]                                                                 |[x1 * (x0 + 1)]
    ['01', '10', '11']                                |[(x1 + 1) * (x0 + 1)]                                                           |[(x1 + 1) * (x0 + 1)]

Next, generate_random_code(d) will generate a random list of codewords in "d" dimension. This is very useful for generating examples to test conjectures::
    sage: generate_random_code(7)
    ['0001101', '1001001', '0011110', '1011101', '0001011', '0011011', '1111101', '0000010', '1011001', '0000001', '1110101', '0100110', '0000101', '0111100', '0100000', '0100011', '0010001',         '1111010', '1101010', '0101011', '1000001', '1100100', '1100110']
    
Building upon that, generate_random_tests(number of tests, dimension) will run "number of tests" amount of comparisons between groebner bases and canonical forms in the specified dimension::
    sage: generate_random_tests(5, 3)
    Differences: 1 Equal bases: 4

    Different Bases:

    Codes                                             |Groebner                                                                        |Canonical
    ['010', '001', '000', '111', '101']               |[(x2 + 1) * x0, x1 * (x0 + x2)]                                                 |[x2 * x1 * (x0 + 1), (x2 + 1) * x0]

    Equal Bases:

    Codes                                             |Groebner                                                                        |Canonical
    ['111']                                           |[x2 + 1, x1 + 1, x0 + 1]                                                        |[x0 + 1, x1 + 1, x2 + 1]
    ['000']                                           |[x2, x1, x0]                                                                    |[x0, x1, x2]
    ['001']                                           |[x2 + 1, x0, x1]                                                                |[x0, x1, x2 + 1]
    ['011', '111', '110']                             |[x1 + 1, (x2 + 1) * (x0 + 1)]                                                   |[x1 + 1, (x2 + 1) * (x0 + 1)]
    
Now, all of these tests use a method called compare_groebner_canonical(gb, cf) which will return a list where the first element is a boolean indicating whether the groebner basis and canonical form equal::
    sage: gb = code.groebner_basis()
    sage: cf = code.canonical()
    sage: compare_groebner_canonical(gb, cf)
    [False, {x1 + x2 + 1, x0*x2}, {x1*x2, x0*x1 + x0, x1*x2 + x1 + x2 + 1, x0*x2}]
    
Another useful method is all_neural_codes(d), which will return a list of all of the possible sets of code words in the specified dimension:
    sage: all_neural_codes(2)
    [[], ['00'], ['01'], ['10'], ['11'], ['00', '01'], ['00', '10'], ['00', '11'], ['01', '10'], ['01', '11'], ['10', '11'], ['00', '01', '10'], ['00', '01', '11'], ['00', '10', '11'], ['01',         '10', '11'], ['00', '01', '10', '11']]
    
Used in is_simplicial(C), support(C) will return the support of a single codeword::
    sage: support('0100011110101')
    [1, 5, 6, 7, 8, 10, 12]

There is also a test suite, assert_build(), which tests whether canonical() will reproduce the results in "The Neural Ring". This method will print an error message if there is an inconsistency.
"""

#*****************************************************************************
#       Copyright (C) 2013 YOUR NAME <your email>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

class NeuralCode:
    r"""

    Class for a neural code, associated to the neural ring.

    >>> assert_build()

    """
    
    def __init__(self, C, order='lex'):
        r"""
        Constructs a neural code.
        
        The Neural Ideal vanishes at every code word in a given list of codes. Many methods have been added to compute traditional objects such as a Groebner Basis, but
        utilizes the BooleanPolynomialRing and generic PolynomialRing's to account for methods not yet implemented in the Boolean Ring, but use the reductive properties (x^2 = x) of
        the Boolean Ring.

        INPUT:

        - ``C`` -- list The list of binary strings representing
            the neural code.

        - ``order`` -- string (default: 'lex') the term order.

        OUTPUT:

            None

        EXAMPLES:

            sage: neural_code = NeuralCode(['000','011'])
            sage: neural_code.canonical()
                  Ideal (x1*x2 + x1, x1*x2 + x2, x0) of Multivariate Polynomial Ring in
                  x0, x1, x2 over Finite Field of size 2
        """
        # throws an error if an empty list was given for the codewords
        if(len(C) == 0):
            print "The collection of codewords must be nonempty\n"
            raise RuntimeError
            
        # temporarily store the dimension for further assertions
        dimension = len(C[0])
        
        # throws an error if the collection of code words is not a list
        if "list" not in str(type(C)):
            print "The collection of codewords must be a list: \n" + "Input: " + str(C) + "\nCorrect example: C = ['001','010'] \n"
            raise TypeError
        
        # throws an error if the elements of the list are not strings
        for i in range(len(C)):
            if not "0" in C[i] and not "1" in C[i]:
                print "Each code word must be comprised of 0's or 1's\n"
                raise TypeError
                
            if(len(C[i]) != dimension):
                print "Each code word must be of the same length, the dimension: " + str(dimension) + "\n"
                raise TypeError
                
            if 'str' not in str(type(C[i])):
                print "Code words must be strings: \n" + "Input: " + str(C[i]) + "\n" + "Correct example: C[0] = '001'\n"
                raise TypeError
            
        self.Codes = C
        self.d = dimension
        
        self.F = PolynomialRing(GF(2), dimension, "x", order=order)
        self.x = self.F.gens()
        
        self.Boolean_Ring = BooleanPolynomialRing(dimension, "x", order=order)
        self.b = self.Boolean_Ring.gens()
        
        self.V = ["".join(x) for x in itertools.product("01", repeat=dimension)]  # generates the possible configurations of code words
        for i in range(len(C)):
            if C[i] in self.V:
                self.V.remove(C[i])
    
    def neural_ideal(self):
        r"""
        Fetches the neural ideal.

        INPUT:

            None

        OUTPUT:

            An ideal in the integer mod 2 ring.

        EXAMPLES:

            sage: neural_code = NeuralCode(['000','011'])
            sage: neural_code.neural_ideal()
                  Ideal (x0*x1*x2 + x0*x2 + x1*x2 + x2, x0*x1*x2 + x0*x1 + x1*x2 + x1, x0*x1*x2 + x0*x1 + x0*x2 + x0, x0*x1*x2 + x0*x2, x0*x1*x2 + x0*x1, x0*x1*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2
        TESTS:

        >>> neural_code = NeuralCode(['000','011'])
        >>> neural_code.neural_ideal()
        Ideal (x0*x1*x2 + x0*x2 + x1*x2 + x2, x0*x1*x2 + x0*x1 + x1*x2 + x1, x0*x1*x2 + x0*x1 + x0*x2 + x0, x0*x1*x2 + x0*x2, x0*x1*x2 + x0*x1, x0*x1*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2
        """
        # rho will represent the collection of rho of v that will generate the ideal
        rho = []
        
        # iterate through the points in V
        for i in range(len(self.V)):
            v = self.V[i]
            product = 1
            
            # calculate rho of v for the particular point
            for j in range(len(v)):
                product = product * (1 - Integer(v[j]) - self.x[j])
            rho.append(product)                                                   
        return self.F.ideal(rho)
        
    def _decomposition_product(self, decomp):
        r"""
        Computes the cartesian product of the ideals in the primary decomposition of the neural ideal.

        INPUT:

        - ``decomp`` -- list The list of ideals

        OUTPUT:
        
            The product as a list of ideals.
        """
        product = 1
        for i in range(len(decomp)):
            product = product * decomp[i]
        
        return product
        
    def groebner_fan(self):
        r"""
        Returns the groebner fan, reduced to the Boolean Ring.

        INPUT:

            None

        OUTPUT:

            The groebner fan as a list of ideals.

        EXAMPLES:
            sage: nc = NeuralCode(['00','01'])
            sage: nc.groebner_fan()
            [Ideal (x0) of Multivariate Polynomial Ring in x0, x1 over Finite Field of size 2]

        TESTS:

        >>> nc = NeuralCode(['00','01'])
        >>> nc.groebner_fan()
        [Ideal (x0) of Multivariate Polynomial Ring in x0, x1 over Finite Field of size 2]

        """
        if len(self.Codes) == 2 ** self.d:
            return [0]
        
        gf = self.neural_ideal().groebner_fan()
        
        original_bases = gf.reduced_groebner_bases()
        
        reduced_bases = []
        
        # iterates through the elements in the reduced groebner bases, maps into boolean ring, then places nonzero elements back into the original ring
        for i in range(len(original_bases)):
            temp = []
            for j in range(len(original_bases[i])): 
                temp_element = self.Boolean_Ring(original_bases[i][j])
                if(temp_element != 0):
                    temp.append(temp_element)
            if len(temp) != 0:
                if temp not in reduced_bases:
                    reduced_bases.append(Set(temp).list())
                    
        for i in range(len(reduced_bases)):
            reduced_bases[i] = self.F.ideal(reduced_bases[i])
        return Set(reduced_bases).list()
        
    def universal_groebner_basis(self):
        r"""
        Returns the universal groebner basis.
        
        Takes the reduced groebner bases, maps their elements to the Boolean Ring, and places all unique elements into a list. 
        This list then constructs an ideal in the polynomial ring mod 2, which is returned.

        INPUT:

            None

        OUTPUT:

            The reduced groebner basis as an ideal.

        EXAMPLES:

            sage: nc = NeuralCode(['001','010','110','001'])
            sage: nc.universal_groebner_basis()
            Ideal (x1 + x2 + 1, x0*x1 + x0, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2

        TESTS:

        >>> nc = NeuralCode(['001','010','110','001'])
        >>> nc.universal_groebner_basis()
        Ideal (x1 + x2 + 1, x0*x1 + x0, x0*x2) of Multivariate Polynomial Ring in x0, x1, x2 over Finite Field of size 2

        """
        if len(self.Codes) == 2 ** self.d:
            return [0]
        
        gf = self.neural_ideal().groebner_fan()
        
        original_bases = gf.reduced_groebner_bases()
        
        universal = []
        
        # iterates through the elements in the reduced groebner bases, maps into boolean ring, then places nonzero elements back into the original ring
        for i in range(len(original_bases)):
            for j in range(len(original_bases[i])): 
                temp_element = self.Boolean_Ring(original_bases[i][j])
                if(temp_element != 0):
                    if temp_element not in universal:
                        universal.append(temp_element)
                    
        return self.F.ideal(Set(universal).list())
        
    def _parse_decomposition(self, decomp):
        r"""Returns the prime ideals that make up the given decomposition."""
        
        primes = []
        for i in range(len(decomp)):
            primes.append(self.F.ideal(decomp[i].gens()))
        return primes
        
    def _booleanIdealReduce(self, M):
        r"""Maps the generators of a given ideal into the Boolean Ring and returns the ideal of the remaining nonzero generators."""
        original = M.gens()
        reduced = []
        
        for i in range(len(original)):
            temp = self.Boolean_Ring(original[i])
            if(temp != 0):
                reduced.append(temp)

        return self.F.ideal(reduced)
        
    def _booleanRingReduce(self, M):
        r"""Maps the generators of a given ideal into the Boolean Ring and returns the ideal of the remaining nonzero generators."""
        original = M
        reduced = []
        
        for i in range(len(original)):
            temp = self.Boolean_Ring(original[i])
            if(temp != 0):
                reduced.append(temp)

        return reduced
        
    def _reduce(self, reduced_decomp_product):
        r"""Returns an ideal whose generators are not multiples of each other. """
        bases = Set(reduced_decomp_product.gens())
        reduced_bases = []
        
        for i in range(len(bases)):
            is_divisible = false
            for j in range(len(bases)):
                r = bases[i].quo_rem(bases[j])
                if (r[1] == 0 and i != j):
                    is_divisible = true
                    break
            if(is_divisible == false):
                reduced_bases.append(bases[i])
        return self.F.ideal(reduced_bases)
        
    def factored_canonical(self, algorithm = "iterative", decomposition_algorithm="gtz"):
        r"""
        Returns the canonical form of the neural ideal, where the generators are factored.

        INPUT:

            None

        OUTPUT:

            The canonical form in which its generators are factored as an 

        EXAMPLES:
            sage: nc = NeuralCode(['0010','0110','1001','1010','1111'])
            sage: nc.factored_canonical()
            [(x2 + 1) * x1, (x3 + 1) * (x2 + 1), x3 * (x0 + 1), (x3 + 1) * x1 * x0, (x2 + 1) * (x0 + 1), x3 * x2 * (x1 + 1)]

        TESTS:

        >>> nc = NeuralCode(['0010','0110','1001','1010','1111'])
        >>> nc.factored_canonical(algorithm="usual")
        [(x2 + 1) * x1, (x3 + 1) * (x2 + 1), x3 * (x0 + 1), (x3 + 1) * x1 * x0, (x2 + 1) * (x0 + 1), x3 * x2 * (x1 + 1)]

        """
        a=[]
        if(self.canonical() == "Empty"):
            return "Empty"
        m=self.canonical(algorithm, decomposition_algorithm).gens()
        for i in range(len(m)):
            a.append(m[i].factor())
        return a
                         
    def canonical(self, algorithm = "iterative", decomposition_algorithm = "pm", threading=False, threads = 2):
        r"""
        Return the canonical form of the neural code's ideal.

        INPUT:

        - ``algorithm`` -- string (default: 'iterative') the algorithm to be used for computing the canonical form
        
        - ``decomposition_algorithm`` -- string (default: 'pm') the algorithm to be used for primary decomposition

        - ``threading`` -- boolean (default: False) indicator to use threading for large computations
        
        - ``threads``   -- integer (default: 2) the number of threads to use

        OUTPUT:

            The canonical form of the neural code as an ideal.

        EXAMPLES:
            sage: C = ['000111','101010','111000','111001','100110']
            sage: nr = NeuralCode(C)
            sage: nr.canonical()
            Ideal (x0*x1 + x1, x1*x2*x5 + x2*x5, x1*x3, x2*x3 + x2 + x3 + 1, x0*x2 + x2, x0*x3*x5, x3*x4*x5 + x4*x5, x2*x3, x0*x4*x5, x2*x4 + x2 + x4 + 1, x1*x3*x5 + x1*x5 + x3*x5 + x5, x0*x4 + x0 + x4 + 1, x1*x2 + x1, x1*x4 + x1 + x4 + 1, x1*x4, x0*x1*x5 + x0*x5, x0*x2*x5 + x0*x5, x3*x4 + x3, x0*x3 + x0 + x3 + 1, x0*x5 + x0 + x5 + 1, x2*x4*x5) of Multivariate Polynomial Ring in x0, x1, x2, x3, x4, x5 over Finite Field of size 2
            

        .. NOTE::

            For very large computations (dimension greater than 8), it may be faster to use threading.

        TESTS:

        >>> C = ['000111','101010','111000','111001','100110']
        >>> nr = NeuralCode(C)
        >>> expected = nr.canonical(algorithm="usual", decomposition_algorithm="gtz")
        >>> iterative_canonical = nr.canonical(algorithm="iterative")
	>>> expected == iterative_canonical
	True
        """
        if (algorithm == "iterative"):
            return iterate_canonical(self.Codes, self.F)
        else:
            # condition for when Groebner = Canonical
            if len(self.Codes) == 1 or len(self.Codes) == (2**self.d - 1) or is_simplicial(self.Codes):
                return self.groebner_basis()

            # calculates an ideal in the traditional way, using the union of varieties
            if (len(self.Codes) == 2 ** self.d):
                return "Empty"

            if self.d >= 8 and len(self.Codes) < 2**(self.d - 2):
                j_c = self.traditional_neural_ideal()
            else:
                j_c = self.neural_ideal()

            # recover the primary decomposition of the ideal J
            if (decomposition_algorithm == "pm"):
                primes = pm_primary_decomposition(j_c)
            else:
                primes = j_c.primary_decomposition(decomposition_algorithm)

            # gets the list of decomposition polynomials
            decomp = self._parse_decomposition(primes)

            # compute the product of the decomposition ideals
            M = self._decomposition_product(decomp)

            ########## Method using threading #########
            if threading:

                # splits the list of products into one for each thread
                splitM = list(self._chunks(M, int(math.ceil(len(M.gens()) / threads))))

                # initializes a thread pool
                pool = ThreadPool(processes=threads)

                # this list will hold all of the threads
                async = []

                # iterate through and start the threads
                for i in range(threads):
                    async.append(pool.apply_async(self._booleanRingReduce, ([splitM[i]])))

                # list to hold the final reduced polynomials
                whole = []

                # get all of the sublists and concatenate them
                for j in range(threads):
                    whole = whole + async[j].get()

                # create the reduced ideal
                threadBooleanReduceM =  self.F.ideal(whole)

                if(len(threadBooleanReduceM.gens()) == 1 and threadBooleanReduceM.gens()[0] == 0):
                    return "Empty"

                # take out multiples
                canonicalForm = self._reduce(threadBooleanReduceM)

                return canonicalForm
            ###########################################

            ### Method using sequential computation ###

            # using Boolean Ring to reduce M
            booleanReduceM = self._booleanIdealReduce(M)

            # case for the zero ideal
            if(len(booleanReduceM.gens()) == 1 and booleanReduceM.gens()[0] == 0):
                return "Empty"

            # reduce again by taking out all generators that are multiples of each other and return
            canonicalForm = self._reduce(booleanReduceM)

            return canonicalForm
        
    def traditional_neural_ideal(self):
        r"""
        Constructs the neural ideal by using the union of varieties.

        INPUT:

            None

        OUTPUT:

            The neural ideal.

        EXAMPLES:
            sage: C = ['11','10','00','01']
            sage: nr = NeuralCode(C)
            sage: nr.traditional_neural_ideal()
            Ideal (x0^4 + x0^2, x0^3*x1 + x0^3 + x0*x1 + x0, x0^3*x1 + x0*x1, x0^2*x1^2 + x0^2*x1 + x1^2 + x1, x0^3*x1 + x0^2*x1, x0^2*x1^2 + x0^2*x1 + x0*x1^2 + x0*x1, x0^2*x1^2 + x0*x1^2, x0*x1^3 + x0*x1^2 + x1^3 + x1^2, x0^3*x1 + x0^3 + x0^2*x1 + x0^2, x0^2*x1^2 + x0^2 + x0*x1^2 + x0, x0^2*x1^2 + x0^2*x1 + x0*x1^2 + x0*x1, x0*x1^3 + x0*x1 + x1^3 + x1, x0^2*x1^2 + x0^2*x1, x0*x1^3 + x0*x1, x0*x1^3 + x0*x1^2, x1^4 + x1^2) of Multivariate Polynomial Ring in x0, x1 over Finite Field of size 2
        

        TESTS:
        >>> C = ['11','10','00','01']
        >>> nr = NeuralCode(C)
        >>> nr.traditional_neural_ideal()
        Ideal (x0^4 + x0^2, x0^3*x1 + x0^3 + x0*x1 + x0, x0^3*x1 + x0*x1, x0^2*x1^2 + x0^2*x1 + x1^2 + x1, x0^3*x1 + x0^2*x1, x0^2*x1^2 + x0^2*x1 + x0*x1^2 + x0*x1, x0^2*x1^2 + x0*x1^2, x0*x1^3 + x0*x1^2 + x1^3 + x1^2, x0^3*x1 + x0^3 + x0^2*x1 + x0^2, x0^2*x1^2 + x0^2 + x0*x1^2 + x0, x0^2*x1^2 + x0^2*x1 + x0*x1^2 + x0*x1, x0*x1^3 + x0*x1 + x1^3 + x1, x0^2*x1^2 + x0^2*x1, x0*x1^3 + x0*x1, x0*x1^3 + x0*x1^2, x1^4 + x1^2) of Multivariate Polynomial Ring in x0, x1 over Finite Field of size 2
        
        """
        poly = []
        
        for i in range(len(self.Codes)):
            ideal = []
            
            for j in range(len(self.Codes[i])):
                neuron = self.Codes[i][j]
                
                if neuron == '1':
                    ideal.append(self.x[j] + 1)
                else:
                    ideal.append(self.x[j])
            poly.append(self.F.ideal(ideal))
        return self._decomposition_product(poly)
        
    def _sigma_tau(self):
        r"""Returns the sets sigma and tau, where sigma is the list of Receptive Field sets whose intersections are subsets of the union of the Receptive Field sets in Tau."""
        list = self.factored_canonical()
        
        if (list == "Empty"):
            return "Empty"
        all_sigma = []
        all_tau = []
        for i in range(len(list)):
            current = str(list[i])
            split = current.split('*')
            sigma = []
            tau = []
            for j in range(len(split)):
                factor = split[j].strip()
                if('(' in factor):
                    tau.append(factor[2])
                else:
                    if ('+' in factor):
                        tau.append(factor[1])
                    else:
                        sigma.append(factor[1])
            all_sigma.append(sigma)
            all_tau.append(tau)
        return (all_sigma , all_tau)
        
    def groebner_basis(self):
        r"""
        Returns the groebner basis of the neural ideal using the libsingular:std algorithm.

        INPUT:

            None

        OUTPUT:

            The groebner basis of the neural code's ideal

        EXAMPLES:
            sage: C = ['01010','11100','11110','01011']
            sage: nr = NeuralCode(C)
            sage: nr.groebner_basis()
            Ideal (x0 + x2, x1 + 1, x2*x3 + x2 + x3 + 1, x2*x4, x3*x4 + x4) of Multivariate Polynomial Ring in x0, x1, x2, x3, x4 over Finite Field of size 2

        TESTS:

        >>> C = ['01010','11100','11110','01011']
        >>> nr = NeuralCode(C)
        >>> nr.groebner_basis()
        Ideal (x0 + x2, x1 + 1, x2*x3 + x2 + x3 + 1, x2*x4, x3*x4 + x4) of Multivariate Polynomial Ring in x0, x1, x2, x3, x4 over Finite Field of size 2

        """
        ni = self.neural_ideal()
        
        gb = ni.groebner_basis(algorithm='libsingular:std')
        
        reduced = []
        
        for i in range(len(gb)):
            if self.Boolean_Ring(gb[i]) != 0:
                reduced.append(gb[i])
                
        if (len(gb) == 1 and gb[0] == 0):
            return "Empty"
        
        return self.F.ideal(reduced)

    def canonical_RF_structure(self):
        r"""
        Prints the Receptive Field structure using the canonical form of the neural ideal.

        INPUT:

            None

        OUTPUT:

            None

        EXAMPLES:
            sage: C = ['110','100','000','010']
            sage: nr = NeuralCode(C)
            sage: nr.canonical_RF_structure()
            Intersection of U_['2'] is empty

        TESTS:

        >>> C = ['110','100','000','010']
        >>> nr = NeuralCode(C)
        >>> nr.canonical_RF_structure()
        Intersection of U_['2'] is empty

        """
        list = self._sigma_tau()
        
        if (list == "Empty"):
            print "Zero ideal : Empty"
            return
            
        sigma = list[0]
        tau = list [1]
        for i in range(len(sigma)):
            if(len(tau[i]) == 0):
                print("Intersection of U_" + str(sigma[i]) + " is empty")
            elif(len(sigma[i]) == 0):
                print("X = Union of U_" + str(tau[i]))
            else:
                print("Intersection of U_" + str(sigma[i]) + " is a subset of Union of U_" + str(tau[i]))

    def _chunks(self, ideal, n):
        r"""Returns a list of n-sized sublists of the given ideal's basis."""
        for i in xrange(0, len(ideal.gens()), n):
            yield ideal.gens()[i:i+n]


def assert_build(algorithm="usual", decomposition_algorithm="pm"):
    """  Asserts that the canonical form calculations give the same results as those in "The Neural Ring" by Curto et al. Used in doctests.  """
    
    
    paper_example_A = [['000','100','010','001','110','101','011','111'],['000','100','010','110','101','111'],['000','100','010','001','110','101','111'],['000','100','010','110','101','011','111'],['000','100','010','110','111'],['000','100','110','101','111'],['000','100','010','101','111'],['000','100','010','001','110','111'],['000','100','001','110','011','111'],['000','100','010','101','011','111'],['000','100','110','101','011','111'],['000','100','110','111'],['000','100','010','111'],['000','100','010','001','111'],['000','110','101','011','111'],['000','100','011','111'],['000','110','101','111'],['000','100','111'],['000','110','111'],['000','111']]
    
    paper_example_B = [['000','100','010','001','110','101'],['000','110','010','110','101'],['000','100','010','101','011'],['000','100','110','101'],['000','100','110','011'],['000','110','101']]
    
    paper_example_C = [['000','100','010','001','110'],['000','100','010','101'],['000','100','011']]
    
    paper_example_D = [['000','100','010','001']]
    
    paper_example_E = [['000','100','010','001','110','101','011'],['000','100','010','110','101','011'],['000','100','110','101','011'],['000','110','011','101']]
    
    paper_example_F = [['000','100','010','110'],['000','100','110'],['000','110']]
    
    paper_example_G = [['000','100']]
    
    paper_example_H = [['000']]
    
    paper_example_I = [['000','100','010']]
    
    all_paper = paper_example_A + paper_example_B + paper_example_C + paper_example_D + paper_example_E + paper_example_F + paper_example_G + paper_example_H + paper_example_I
    
    F = PolynomialRing(GF(2), 3, "x")
    x = F.gens()
        
    expected_ideals = [[0], [x[2] * (x[0] + 1)], [x[2] * x[1] * (x[0] + 1)], [x[2] * (x[1] + 1) * (x[0] + 1)], [x[2] * (x[1] + 1), x[2] * (x[0] + 1)], [x[2] * (x[0] + 1), x[1] * (x[0] + 1)], [x[2] * (x[0] + 1), (x[2] + 1) * x[1] * x[0]], [x[2] * (x[1] + 1) * x[0], x[2] * x[1] * (x[0] + 1)], [x[2] * (x[1] + 1) * x[0], (x[2] + 1) * x[1] * (x[0] + 1)], [x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * x[0]], [x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * (x[0] + 1)], [x[2] * (x[1] + 1), x[2] * (x[0] + 1), x[1] * (x[0] + 1)], [x[2] * (x[1] + 1), x[2] * (x[0] + 1), (x[2] + 1) * x[1] * x[0]], [x[2] * (x[1] + 1) * x[0], (x[2] + 1) * x[1] * x[0], x[2] * x[1] * (x[0] + 1)], [(x[2] + 1) * (x[1] + 1) * x[0], x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * (x[0] + 1)], [x[2] * (x[1] + 1), (x[2] + 1) * x[1]], [(x[2] + 1) * (x[1] + 1) * x[0], x[2] * (x[0] + 1), x[1] * (x[0] + 1)], [x[2] * (x[1] + 1), (x[2] + 1) * x[1], x[2] * (x[0] + 1), x[1] * (x[0] + 1)], [x[2] * (x[1] + 1), (x[1] + 1) * x[0], x[2] * (x[0] + 1), x[1] * (x[0] + 1)], [(x[2] + 1) * x[1], x[2] * (x[0] + 1), x[1] * (x[0] + 1), (x[1] + 1) * x[0], x[2] * (x[1] + 1), (x[2] + 1) * x[0]], [x[2] * x[1]], [x[2] * x[1], x[2] * (x[0] + 1), (x[2] + 1) * (x[1] + 1) * x[0]], [x[2] * (x[1] + 1) * (x[0] + 1), x[1] * x[0]], [x[2] * x[1], x[2] * (x[0] + 1), x[1] * (x[0] + 1)], [x[2] * (x[1] + 1), x[2] * x[0], (x[2] + 1) * x[1] * (x[0] + 1)], [(x[2] + 1) * (x[1] + 1) * x[0], x[2] * (x[0] + 1), x[2] * x[1], x[1] * (x[0] + 1)], [x[2] * x[0], x[2] * x[1]], [x[2] * x[1], x[2] * (x[0] + 1), x[1] * x[0]], [x[2] * (x[1] + 1), (x[2] + 1) * x[1], x[2] * x[0], x[1] * x[0]], [x[1] * x[0], x[2] * x[0], x[2] * x[1]], [x[2] * x[1] * x[0]], [x[2] * (x[1] + 1) * (x[0] + 1), x[2] * x[1] * x[0]], [x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * (x[0] + 1), x[2] * x[1] * x[0]], [(x[2] + 1) * (x[1] + 1) * x[0], x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * (x[0] + 1), x[2] * x[1] * x[0]], [x[2]], [x[2], x[1] * (x[0] + 1)], [x[2], (x[1] + 1) * x[0], x[1] * (x[0] + 1)], [x[1], x[2]], [x[0], x[1], x[2]], [x[1] * x[0], x[2]], [x[2] * (x[1] + 1) * (x[0] + 1), x[2] * x[1] * x[0]], 
[x[2] * (x[1] + 1), x[2] * x[0], (x[2] + 1) * x[1] * (x[0] + 1)], [x[2] * (x[1] + 1) * (x[0] + 1), x[1] * x[0]], [x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * (x[0] + 1), x[2] * x[1] * x[0]], [(x[2] + 1) * (x[1] + 1) * x[0], x[2] * (x[1] + 1) * (x[0] + 1), (x[2] + 1) * x[1] * (x[0] + 1), x[2] * x[1] * x[0]]]

    for i in range(len(all_paper)):
            neural_ideal = NeuralCode(all_paper[i])
            canonical_form = neural_ideal.canonical(algorithm, decomposition_algorithm)

            if (canonical_form == "Empty" and expected_ideals[i] == [0]):
                continue

            if (canonical_form == "Empty" and expected_ideals[i] != [0]):
                print "Build failed on Neural Code " + str(all_paper[i]) + " Expected: " + str(expected_ideals[i]) + " but was " + str(gens) + ". Decomposition = " + decomp_algs[k] + "\n"
                break

            gens = canonical_form.gens()
            
            if (len(gens) != len(expected_ideals[i])):
                print "Build failed on Neural Code " + str(all_paper[i]) + " Expected: " + str(expected_ideals[i]) + " but was " + str(gens) + ". Decomposition = " + decomp_algs[k] + "\n"
                break
            for j in range(len(gens)):
                if (gens[j] not in expected_ideals[i]):
                    print "Build failed on Neural Code " + str(all_paper[i]) + " Expected: " + str(expected_ideals[i]) + " but was " + str(gens) + ". Decomposition = " + decomp_algs[k] + "\n"
                    break


def generate_random_code(dimension, num_code_words=None):
    r"""
    Generates a random list of binary strings (code words) in the specified dimension.

    INPUT:

    - ``dimension`` -- integer the dimension of the boolean ring (the length of the binary strings).
    - ``num_code_words`` -- optional integer the number of code words.

    OUTPUT:

    The code words as a list of strings.

    EXAMPLES:
        sage: generate_random_code(5)
        ['10000', '10001', '01011', '11011', '10011', '10111', '11001', '11000', '01100', '11100', '11101', '11010', '01010', '00110', '00001', '00010', '00011', '00100']
    """
    all_combinations = ["".join(x) for x in itertools.product("01", repeat=dimension)]      # generates all possible combinations of 0 and 1 of length dimension
    code_words = []
    if num_code_words is None:
        num_code_words = randint(1, 2 ** dimension)

    while len(code_words) != num_code_words:
        code_words.append(all_combinations[randint(0, 2 ** dimension - 1)])                 # randomly select from all combinations and add to code words
        code_words = Set(code_words).list()
    return code_words


def all_neural_codes(dimension):
    r"""
    Returns a list of all of the neural codes in the specified dimension.

    INPUT:

    - ``dimension`` -- integer the dimension of the boolean ring.

    OUTPUT:

    All neural codes as a list of lists of strings.
    
    EXAMPLES:
        sage: all_neural_codes(2)
        [[], ['00'], ['01'], ['10'], ['11'], ['00', '01'], ['00', '10'], ['00', '11'], ['01', '10'], ['01', '11'], ['10', '11'], ['00', '01', '10'], ['00', '01', '11'], ['00', '10', '11'], ['01','10', '11'], ['00', '01', '10', '11']]

    TESTS:

    >>> all_neural_codes(2)
    [[], ['00'], ['01'], ['10'], ['11'], ['00', '01'], ['00', '10'], ['00', '11'], ['01', '10'], ['01', '11'], ['10', '11'], ['00', '01', '10'], ['00', '01', '11'], ['00', '10', '11'], ['01', '10', '11'], ['00', '01', '10', '11']]

    """
    all = ["".join(x) for x in itertools.product("01", repeat=dimension)]
    return Combinations(all).list()


def support(C):
    r"""
    Return the support of the code word.

    INPUT:

    - ``C`` -- string The code word.

    OUTPUT:

    The support as a list.

    EXAMPLES:
        sage: C = '0101'
        sage: support(C)
        [1, 3]

    TESTS:

    >>> C = '0101'
    >>> support(C)
    [1, 3]

    """
    support = []
    for i in range(len(C)):
       if C[i] == '1':
           support.append(i)
    return support


def is_simplicial(C):
    r"""
    Returns a boolean determining if a list of code words is a simplicial code.

    INPUT:

    - ``C`` -- List of binary strings(codes)

    OUTPUT:

    True if the code is a simplicial code, false otherwise.

    EXAMPLES:
        sage: is_simplicial(['00','01','10','11'])
        True
        sage: is_simplicial(['000','101','010','111'])
        False

    TESTS:

    >>> is_simplicial(['00','01','10','11'])
    True
    >>> is_simplicial(['000','101','010','111'])
    False

    """
    max = 0
    index = 0
    
    for i in range(len(C)):
        if len(support(C[i])) > max:
            index = i
            
    count = 0
    maxSupport = support(C[index])
    for j in range(len(C)):
        supp = support(C[j])
        
        if len(supp) == 0:
            count = count + 1
        else:
            subset = True
            for k in range(len(supp)):
                if supp[k] not in maxSupport:
                    subset = False
            if subset:
                count = count + 1
    if count == 2 ** len(maxSupport):
        return True
    return False


def pm_primary_decomposition(IDEAL):
    #step 1 initialization step
    final = []
    P = []
    Q = []
    D = [IDEAL]
    new_D = D
    
    #step 5 recursion step (which is a loop of steps 2-4)
    while D <> []:
        [D, Q] = D_Q(D, Q)
    
    #step 6 final step
    P = list(set(Q))
    for m in range(len(P)):
        final.extend(reduced_primes_list(P, m))
    return final


#Step 6: If one ideal contains the generators of another ideal in the list, then it is redundant since we are taking the intersection of these ideals. This command gets a list of the non redundant ideals
def reduced_primes_list(P, m):
    plist = []
    contains = False
    for n in range(len(P)):
        if m == n:
            continue
        is_in = True
        for o in range(len(P[n].gens())):
            if P[n].gens()[o] in P[m].gens():
                continue
            else:
                is_in = False
                break
        if is_in == True:
            contains = True
            break
    if contains == False:
        plist.append(P[m])
    return list(Set(plist))


#steps 2-4
def D_Q(D, Q):
    new_D = []
    
    #creating D_I
    for I in D:
        D_I = []
        
        #finds the first nonlinear generator of I (step 2.1)
        index = 0
        for i in range(len(I.gens())):
            factors = list(I.gens()[i].factor())
            if len(factors) <> 1:
                break
            index = index + 1
            
        #creates list of factors of that first generator (z_i's) called gen_fac_list
        gen_fac = list(I.gens()[index].factor())
        gen_fac_list = []
        for i in range(len(gen_fac)):
            gen_fac_list.append(gen_fac[i][0])
            
        #step 3
        for i in range(len(gen_fac_list)):
            z = gen_fac_list[i]
            D_I.append(reduced_ideal(z, I))
        new_D.extend(D_I)
    
    #step 4
    E = list(Set(new_D))
    D = []
    for k in range(len(E)):
        if is_linear(E[k]):
            if E[k].is_prime():
                Q.append(E[k])
        else:
            D.append(E[k])
    return [D, Q]


#checks if the generators of an ideal are linear by checking if they have only 1 factor
def is_linear(ideal):
    boolean = True
    for i in range(len(ideal.gens())):
        factors = list(ideal.gens()[i].factor())
        if len(factors) != 1:
            boolean = False
            break
    return boolean


#executes the reduction in step 3 for a given ideal and given z
def reduced_ideal(z, I):
    L = []
    M = []
    N = []

    #creates a list of non reduced generators 
    for j in range(len(I.gens())):
        L.append(I.gens()[j])

    #implements the condition z = 0
    for j in range(len(L)):
        if z.divides(L[j]):
            continue
        else:
            M.append(L[j])

    #implements the condition z+1 = 1
    for j in range(len(M)):
        if (z+1).divides(M[j]):
            N.append(M[j].quo_rem(z+1)[0])
        else:
            N.append(M[j])

    #sorts list to avoid duplicates and returns the ideal generated by the list
    N.append(z)
    N_1 = sorted(N)
    return Ideal(N_1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

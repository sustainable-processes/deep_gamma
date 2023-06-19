import sqlite3
import pandas as pd
import pathlib
from COSMOpy import CosmoPy, Mixture
from tqdm import tqdm
import platform as p


def guess_cosmobase_path():
    system = p.system()
    if system == "Darwin":
        base_path = pathlib.Path("/Applications/BIOVIA")
    elif system == "Linux":
        base_path = pathlib.Path.home() / "BIOVIA"

    path = (
        base_path
        / "COSMOtherm2020/COSMOtherm/DATABASE-COSMO/COSMObase2020/BP-TZVPD-FINE.db"
    )
    return path


def get_cosmo_search_df(path=None):
    """Get COSMO search df

    path is the path to the database
    """
    if path is None:
        path = guess_cosmobase_path()
    db_path = pathlib.Path(path)

    # Connect to SQLite3
    conn = sqlite3.connect(db_path)

    # Read into dataframe
    df = pd.read_sql("SELECT * FROM SearchResultSet", conn)

    # Close SQLite3 connection
    conn.close()

    return df


class CosmoCalculate:
    CAS = "CAS"
    UNICODE = "UNICODE"
    NAME = "NAME"
    SMILES = "SMILES"

    """  A convencience wrapper around CosmoPy
    
    Parameters
    ---------- 
    calculation_name : str
        Name of the calculation. For example "Activity Coefficient".
    lookup_name : str
        Name of the column for molecule identifiers in the Pandas series 
        passed in at call time. Defaults to "casNumber".
    lookup_type : str, optional
        Identifier used for looking up molecules. Defaults to CosmoCalculate.CAS but
        can also be UNICODE, NAME, or SMILES.
    level : str, optional
        Level for running calculations. Defaults to TZVPD-FINE.
    n_cores : int, optional
        Number of cores to use for calculations in total. Defaults to 1.
    cores_per_job : int, optional
        Number of cores to use for each job. Defaults to 1.
    
    
    Examples
    --------
    >>> calc_func = CosmoCalculate("Boiling Point",  \
                                   "Tboil", \
                                   lookup_name="uniqueCode12", \
                                   lookup_type=CosmoCalculate.UNICODE, \
                                   background=True, \
                                   n_cores=1)
    >>> df = get_cosmo_search_df()
    >>> mols = [calc_func(row) for _, row in  \
                df.iloc[rows_read:rows_read + batch_size].iterrows()]
    
    """

    def __init__(self, calculation_name, **kwargs):
        self.calculation_name = calculation_name
        self.property_name = kwargs.get("property_name")
        self.lookup_name = kwargs.get("lookup_name", "casNumber")
        self.lookup_type = kwargs.get("lookup_type", self.CAS)

        self.search_df = get_cosmo_search_df()

        # Set up cosmo-rs
        self.ct = CosmoPy().cosmotherm
        self.ct.setLevel(kwargs.get("level", "TZVPD-FINE"))
        n_cores = kwargs.get("n_cores", 1)
        self.ct.setUseCores(kwargs.get("cores_per_job", 1))
        if n_cores == "max":
            n_cores = self.ct.getFreeCores()
        self.ct.setNCores(int(n_cores))

        # Create scracth directory if it doesn't already exist
        pathlib.Path("scratch/").mkdir(exist_ok=True)
        self.ct.setScratch("scratch/")

        # Calculate in background. Need to manage queue
        self.background = kwargs.get("background", False)

    def __call__(self, *rows, **kwargs):
        """Calculate properties using COSMOtherm

        Parameters
        ----------
        row : pd.Series
            Row of pandas dataframe that has a casNumber column
        xmin : float
            Minimum of composition grid for binary mixtures
        xmax : float
            Maximum of composition grid for binary mixtures
        xstep : float
            Step of composition grid for binary mixtures
        Tmin : float
            Minimum of temperature grid for binary mixtures
        Tmax : float
            Maximum of temperature grid for binary mixtures
        Tstep : float
            Step of temperature grid for binary mixtures

        Returns
        -------
        Row with property value inserted

        """
        mols = []
        for row in rows:
            value = row[self.lookup_name]
            mol = self.lookup_molecule(value)
            if mol is None:
                raise ValueError(f"No molecule found for {value}")
            mols.append(mol)
        # Run calculations for a single molecule
        if len(mols) == 0:
            self.ct.define(mols[0], self.calculation_name)
            self.ct.compute(use_scratch=True, background=self.background)
        # Run calculation for mixture
        else:
            # Create mixture
            mixture_name = "".join(
                [f"{row[self.lookup_name]}__" for row in rows]
            ).rstrip("__")
            mixture = Mixture(mixture_name)
            for mol in mols:
                mixture.addCompound(mol)

            # Define calculation
            binary_check_1 = (
                kwargs.get("xmin") or kwargs.get("xlist") or kwargs.get("Tmin")
            )
            binary_check_2 = len(mols) == 2
            if binary_check_1 and binary_check_2:
                self.ct.defineBinaryGrid(
                    mixture=mixture, method=self.calculation_name, **kwargs
                )
            # Simple mixture calculation
            else:
                self.ct.define(mixture, self.calculation_name, **kwargs)

            # Schedule comptuation
            self.ct.compute(use_scratch=True, background=self.background)
        return mols

    def lookup_molecule(self, value):
        if self.lookup_type == self.CAS:
            row = self.search_df[self.search_df["casNumber"] == value]
            if not row.empty:
                value = str(row["uniqueCode12"].values[0])
                mol = self.ct.searchUnicode(value)[0]
            else:
                mol = None
        elif self.lookup_type == self.UNICODE:
            mol = self.ct.searchUnicode(value)[0]
        elif self.lookup_type == self.NAME:
            row = self.search_df[self.search_df["compoundName"] == value]
            value = str(row["uniqueCode12"].values[0])
            mol = self.ct.searchUnicode(value)[0]
        elif self.lookup_type == self.SMILES:
            row = self.search_df[self.search_df["smiles"] == value]
            value = str(row["uniqueCode12"].values[0])
            mol = self.ct.searchUnicode(value)[0]
        else:
            raise ValueError(f"Unknown lookup type: {self.lookup_type}.")
        return mol

    # def read_result(self, row, **kwargs):
    #     value = row[self.lookup_name]
    #     mol = self.lookup_molecule(value)
    #     return mol.getProperty(self.property_name, **kwargs)

import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import os
from aerosandbox.aerodynamics.aero_2D.xfoil import XFoil
import matplotlib.pyplot as plt


def CL_range(design_CLs, num_points):
    """
    Generate a range of CL values around a design CL.
    design_CLs: List of design CL values.
    num_points: Number of points to generate.
    """
    # Generate a range of CL values
    CL_range = np.linspace(design_CLs[0], design_CLs[1], num_points)
    return CL_range

def weighted_average(data, weights):
    """
    Calculate the weighted average of a list of data points.
    data: List of data points.
    weights: List of weights corresponding to the data points.
    """
    # Calculate the weighted average
    weighted = 0
    for i in range(len(data)):
        if data[i] is None:
            print('some data is None')
            continue
        weighted += np.dot(data[i], weights)
    return weighted / sum(weights)

class AirfoilAnalysis(asb.Airfoil):
    """Class to analyze an airfoil using NeuralFoil or XFOIL. 
    
    Parameters:
    airfoil: str
        Name of the airfoil to analyze.
    cls: list
        List of CL values to analyze.
    reynolds: list
        List of Reynolds numbers to analyze.
    CLweights: list, optional
        List of weights for each CL value. Used for scoring.
    mass: bool, optional.
        If True, airfoil mass is considered in scoring.
    custom_weights: list, optional
        List of custom weights for scoring. Default is [20, 0, 3, 0.2, 5, 5].
    """
    def __init__(self, airfoil, cls, reynolds, CLweights=None, mass=True, custom_weights=None):
        self.airfoil = asb.Airfoil(airfoil)
        self.cls = cls
        self.reynolds = reynolds
        self.cds = []
        self.cms = []
        self.xtr_top = []
        self.xtr_bot = []
        self.maxCL = None
        self.score = 0 
        self.failed=False # True if the airfoil analysis failed
        self.mass = mass # True if airfoil mass is to be considered in scoring
        self.failure_reason = []
        if CLweights is not None:
            self.CLweights = CLweights
        else:
            self.CLweights = [1]*len(cls)
        if custom_weights is not None:
            self.weights = custom_weights
        else:
            self.custom_weights = [20, 0, 3, 0.2, 5, 5]

    def run_xfoil(self):
        raise NotImplementedError("XFOIL is not implemented in this version of the code. Please use NeuralFoil instead.")
        for re in self.reynolds:
            try:
                xfoil = XFoil(
                    airfoil=self.airfoil,
                    Re=re,
                    mach=0,
                    verbose=False,
                    xfoil_repanel=True
                ).cl(cl = self.cls)
            except:
                # print(f'Failed to run XFOIL for airfoil {self.airfoil.name}')
                self.failed = True
            self.cds.append(xfoil['CD'])
            self.cms.append(xfoil['CM'])
            self.xtr_top.append(xfoil['Top_Xtr'])
            self.xtr_bot.append(xfoil['Bot_Xtr'])


    def run_nf(self, min_confidence=0.9):
        """Run NeuralFoil on the airfoil for the given CL values and Reynolds numbers.
        Parameters:
        min_confidence: float, optional
            Minimum confidence value for NeuralFoil analysis. Default is 0.9.
        """
        for re in self.reynolds:
            opti = asb.Opti()
            alpha = opti.variable(init_guess=2, scale=0.1, n_vars=len(self.cls))
            nf = self.airfoil.get_aero_from_neuralfoil(
                alpha=alpha,
                Re=re)
            
            opti.subject_to(nf['CL'] == self.cls)

            opti.minimize(-np.mean(nf['analysis_confidence']))
            try:
                sol = opti.solve(verbose=False)
                nf = sol(nf)
                for i in range(len(self.cls)):
                    if nf['analysis_confidence'][i] < min_confidence:
                        self.failed = True
                        self.failure_reason.append('NF_confidence')
                self.cds.append(nf['CD'])
                self.cms.append(nf['CM'])
                self.xtr_top.append(nf['Top_Xtr'])
                self.xtr_bot.append(nf['Bot_Xtr'])
            except:
                self.failed = True
                self.failure_reason.append('NF_solve')

    def run_nf_cl(self, cl):
        """Run NeuralFoil on the airfoil for the given CL values and Reynolds numbers."""
        opti = asb.Opti()
        alpha = opti.variable(init_guess=2, scale=0.1, n_vars=len(cl))
        nf = self.airfoil.get_aero_from_neuralfoil(
            alpha=alpha,
            Re=self.reynolds[0])
        
        opti.subject_to(nf['CL'] == cl)
        sol = opti.solve(verbose=False)
        nf = sol(nf)
        return nf
    
    def nf_maxCL(self, reynolds=None, min_confidence=0.9):
        """Find the maximum CL for the airfoil using NeuralFoil.

        Parameters:
        reynolds: float, optional
            Reynolds number to analyze. Default is the first value in the reynolds list.
        min_confidence: float, optional
            Minimum confidence value for NeuralFoil analysis. Default is 0.9.
        """
        if reynolds is None:
            reynolds = self.reynolds[0]
        opti = asb.Opti()
        alpha = opti.variable(init_guess=10, scale=0.1, n_vars=1)
        nf = self.airfoil.get_aero_from_neuralfoil(
            alpha=alpha,
            Re=reynolds)
        
        opti.subject_to(nf['analysis_confidence'] > min_confidence)
        opti.minimize(-nf['CL'])
        try:
            sol = opti.solve(verbose=False)
            nf = sol(nf)
            self.maxCL = nf['CL']
        except:
            print(f'Failed to find max CL for airfoil {self.airfoil.name}')
            self.failed = True
            self.failure_reason.append('NF_maxCL')
        # return nf['CL']

    
    def score_result(self, weights=None):
        """Score the airfoil based on the analysis results.

        Parameters:
        weights: list, optional
            List of weights for scoring the airfoil.
        """
        w1, w2, w3, w4, w5, w6 = self.custom_weights
        if weights is None:
            weights = self.CLweights
        score = 0
        for i in range(len(self.cls)):
            cds = self.cds[i]
            cd_weighted_avg = weighted_average(cds, weights)
            score += (
                w1*-cd_weighted_avg/0.00758194 + #normalized by sd7032 CD for simplicity
                w2*np.mean(self.cms[i]) + 
                w3*np.mean(self.xtr_top[i]) + 
                w4*np.mean(self.xtr_bot[i]) +
                0
                )
        if self.maxCL is not None:
            score += w5*self.maxCL
        if self.mass:
            score += -w6*self.airfoil.area()/0.06 #normalized by sd7032 area for simplicity
        self.score = float(score)

class BatchAirfoil():
    def __init__(self, airfoil_path, cls, reynolds, min_thick=0.0, nf=True, maxCL=True, CLweights=None, weight=True, maxCL_Reynolds = 250000, min_confidece=0.9, alivebar=False):
        """Class to analyze a batch of airfoils using NeuralFoil or XFOIL.

        Parameters:
        airfoil_path: str or list
            Path to the folder containing airfoil files, or a list of airfoil files.
        cls: list
            List of CL values to analyze.
        reynolds: list
            List of Reynolds numbers to analyze.
        min_thick: float, optional
            Minimum thickness of the airfoil. Default is 0.
        nf: bool, optional
            If True, NeuralFoil is used for analysis. Default is True.
        maxCL: bool, optional
            If True, the maximum CL value is found. Default is True.
        CLweights: list, optional
            List of weights for each CL value. Used for scoring.
        weight: bool, optional
            If True, airfoil mass is considered in scoring. Default is True.
        maxCL_Reynolds: float, optional
            Reynolds number for takeoff. Default is 250000.
        min_confidence: float, optional
            Minimum confidence value for NeuralFoil analysis. Default is 0.9.
        alivebar: bool, optional
            Display a loading bar. Default is False.

            I'm a sucker for loading bars. This one I designed just for you! 
            However... if you don't want one, you can disable it.
        """
        if isinstance(airfoil_path, str):
            try:
                self.airfoil_files = [f for f in os.listdir(airfoil_path) if f.endswith('.dat')]
            except Exception as e:
                raise Exception(f"Failed to load airfoils from {airfoil_path}: {e}")
        elif isinstance(airfoil_path, list):
            self.airfoil_files = airfoil_path
        self.cls = cls
        self.reynolds = reynolds
        self.min_thick = min_thick
        self.nf = nf
        self.maxCL = maxCL
        self.weight = weight
        self.maxCL_Reynolds = maxCL_Reynolds
        self.min_confidence = min_confidece
        if CLweights is None:
            self.CLweights = [1]*len(cls)

        # DataFrame to store results
        self.results_df = pd.DataFrame(columns=[
            "Name", "CLs", "CDs", "CMs", "Xtr_Top", "Xtr_Bot", "CLmax", "Score", "Failure_Reason", "Airfoil_Object"
        ])

    def run_batch(self):
        """Run the batch analysis on the airfoils."""
        failed = []
        num_failed = 0
        # def run():
        if self.alivebar:
            from alive_progress import alive_bar, animations
            custom = animations.bar_factory(chars='==', tip='☁️✈️', background='⠈⠐⠠⢀⡀⠄⠂⠁', borders = ('🛫→', '←🛬'), errors='💥')
            with alive_bar(len(self.airfoil_files), title="Analyzing airfoils...", bar=custom) as bari:
                for airfoil_file in self.airfoil_files:
                    try:
                        airfoil = AirfoilAnalysis(airfoil_file, self.cls, self.reynolds)
                    except Exception as e:
                        print(f"Failed to load airfoil {airfoil_file}: {e}")
                        num_failed += 1
                        failed.append(airfoil_file)
                        continue

                    # Check thickness
                    try:
                        if airfoil.airfoil.max_thickness() < self.min_thick:
                            print(f"Airfoil {airfoil.airfoil.name} is too thin")
                            airfoil.failure_reason.append("Thin")
                            bari()
                            continue
                    except:
                        print(f"Failed to load airfoil {airfoil.airfoil.name}")
                        num_failed += 1
                        failed.append(airfoil.airfoil.name)
                        bari()
                        continue

                    # Run NeuralFoil or XFOIL
                    if self.nf:
                        airfoil.run_nf()
                        if airfoil.failed:
                            print(f"Failed to run NeuralFoil for airfoil {airfoil.airfoil.name}")
                            num_failed += 1
                            failed.append(airfoil.airfoil.name)
                            bari()
                            continue
                    else:
                        airfoil.run_xfoil()
                        if airfoil.failed:
                            print(f"Failed to run XFOIL for airfoil {airfoil.airfoil.name}")
                            num_failed += 1
                            failed.append(airfoil.airfoil.name)
                            bari()
                            continue

                    # Find CLmax if required
                    if self.maxCL:
                        airfoil.nf_maxCL(reynolds=self.maxCL_Reynolds)
                        if airfoil.failed:
                            print(f"Failed to find max CL for airfoil {airfoil.airfoil.name}")
                            num_failed += 1
                            failed.append(airfoil.airfoil.name)
                            bari()
                            continue

                    # Score the airfoil
                    airfoil.score_result(weights=self.CLweights)

                    # Store results in DataFrame
                    self.results_df.loc[airfoil.airfoil.name] = {
                        "Name": airfoil.airfoil.name,
                        "Airfoil_Object": airfoil.airfoil,  # Storing asb.Airfoil object
                        "CLs": airfoil.cls,
                        "CDs": airfoil.cds,
                        "CMs": airfoil.cms,
                        "Xtr_Top": airfoil.xtr_top,
                        "Xtr_Bot": airfoil.xtr_bot,
                        "CLmax": airfoil.maxCL,
                        "Score": airfoil.score,
                        "Failure_Reason": "; ".join(airfoil.failure_reason) if airfoil.failure_reason else "None"
                    }

                    print(f"Airfoil {airfoil.airfoil.name} analyzed successfully!")
                    bari()
            self.results_df = self.results_df.sort_values(by="Score", ascending=False)
            print(f"Failed to analyze {num_failed} airfoils out of {len(self.airfoil_files)} ({num_failed / len(self.airfoil_files) * 100:.2f}%)")
        else:
            for airfoil_file in self.airfoil_files:
                    try:
                        airfoil = AirfoilAnalysis(airfoil_file, self.cls, self.reynolds)
                    except Exception as e:
                        print(f"Failed to load airfoil {airfoil_file}: {e}")
                        num_failed += 1
                        failed.append(airfoil_file)
                        continue

                    # Check thickness
                    try:
                        if airfoil.airfoil.max_thickness() < self.min_thick:
                            print(f"Airfoil {airfoil.airfoil.name} is too thin")
                            airfoil.failure_reason.append("Thin")
                            bari()
                            continue
                    except:
                        print(f"Failed to load airfoil {airfoil.airfoil.name}")
                        num_failed += 1
                        failed.append(airfoil.airfoil.name)
                        bari()
                        continue

                    # Run NeuralFoil or XFOIL
                    if self.nf:
                        airfoil.run_nf()
                        if airfoil.failed:
                            print(f"Failed to run NeuralFoil for airfoil {airfoil.airfoil.name}")
                            num_failed += 1
                            failed.append(airfoil.airfoil.name)
                            bari()
                            continue
                    else:
                        airfoil.run_xfoil()
                        if airfoil.failed:
                            print(f"Failed to run XFOIL for airfoil {airfoil.airfoil.name}")
                            num_failed += 1
                            failed.append(airfoil.airfoil.name)
                            bari()
                            continue

                    # Find CLmax if required
                    if self.maxCL:
                        airfoil.nf_maxCL(reynolds=self.maxCL_Reynolds)
                        if airfoil.failed:
                            print(f"Failed to find max CL for airfoil {airfoil.airfoil.name}")
                            num_failed += 1
                            failed.append(airfoil.airfoil.name)
                            bari()
                            continue

                    # Score the airfoil
                    airfoil.score_result(weights=self.CLweights)

                    # Store results in DataFrame
                    self.results_df.loc[airfoil.airfoil.name] = {
                        "Name": airfoil.airfoil.name,
                        "Airfoil_Object": airfoil.airfoil,  # Storing asb.Airfoil object
                        "CLs": airfoil.cls,
                        "CDs": airfoil.cds,
                        "CMs": airfoil.cms,
                        "Xtr_Top": airfoil.xtr_top,
                        "Xtr_Bot": airfoil.xtr_bot,
                        "CLmax": airfoil.maxCL,
                        "Score": airfoil.score,
                        "Failure_Reason": "; ".join(airfoil.failure_reason) if airfoil.failure_reason else "None"
                    }

                    print(f"Airfoil {airfoil.airfoil.name} analyzed successfully!")
                    bari()
            self.results_df = self.results_df.sort_values(by="Score", ascending=False)
            print(f"Failed to analyze {num_failed} airfoils out of {len(self.airfoil_files)} ({num_failed / len(self.airfoil_files) * 100:.2f}%)")

    def save_results(self, topN = 10, filename="airfoil_results.csv"):
        """ Saves the results DataFrame to a CSV file. """
        if topN is None:
            topN = self.results_df
        else:
            topN = self.results_df.head(topN)
        topN = topN.drop(columns=["Airfoil_Object"])
        topN.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def draw_analysis(self, topN=10, reynolds=None, save=False):
        if reynolds is None:
            reynolds = self.reynolds[0]
        topN_airfoils = self.results_df.head(topN)
        alphas = np.linspace(-10, 20, 30)
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].set_title('CL vs CD')
        ax[0, 1].set_title('CL vs Alpha')
        ax[1, 0].set_title('CL/CD vs Alpha')
        ax[1, 1].set_title('Xtr vs Alpha')

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (airfoil_name, airfoil_obj) in enumerate(topN_airfoils['Airfoil_Object'].items()):
            nf = airfoil_obj.get_aero_from_neuralfoil(
                alpha=alphas,
                Re=[reynolds]*len(alphas)
            )
            color = color_cycle[i % len(color_cycle)]
            ax[0, 0].plot(nf['CD'], nf['CL'], label=airfoil_name)
            ax[0, 1].plot(alphas, nf['CL'], label=airfoil_name)
            ax[1, 0].plot(alphas, nf['CL']/nf['CD'], label=airfoil_name)
            ax[1, 1].plot(alphas, nf['Top_Xtr'], linestyle="-", color=color, label=f"{airfoil_name} Top")
            ax[1, 1].plot(alphas, nf['Bot_Xtr'], linestyle="--", color=color, label=f"{airfoil_name} Bot")
            
        ax[0, 0].set_xlabel('CD')
        ax[0, 0].set_ylabel('CL')
        ax[0, 1].set_xlabel('Alpha')
        ax[0, 1].set_ylabel('CL')
        ax[1, 0].set_xlabel('Alpha')
        ax[1, 0].set_ylabel('CL/CD')
        ax[1, 1].set_xlabel('Alpha')
        ax[1, 1].set_ylabel('Xtr')
        for a in ax.flat:
            a.grid(True, linestyle="--", alpha=0.6)
        ax[0,0].legend(loc='center right', )
        plt.tight_layout()
        plt.show()

        if save:
            fig.savefig('top_airfoil_comparison.png')

def compare(airfoils, reynolds, save=False):
    """Draw a graph and compare airfoils using NeuralFoil.

    Parameters:
    airfoils: list
        List of Airfoil objects to compare.
    reynolds: float
        Reynolds number to analyze.
    save: bool, optional
        If True, the plot is saved. Default is False.
    """
    alphas = np.linspace(-10, 20, 30)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].set_title('CL vs CD')
    ax[0, 1].set_title('CL vs Alpha')
    ax[1, 0].set_title('CL/CD vs Alpha')
    ax[1, 1].set_title('Xtr vs Alpha')

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, airfoil in enumerate(airfoils):
        nf = airfoil.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=[reynolds]*len(alphas)
        )
        color = color_cycle[i % len(color_cycle)]
        ax[0, 0].plot(nf['CD'], nf['CL'], label=airfoil.name)
        ax[0, 1].plot(alphas, nf['CL'], label=airfoil.name)
        ax[1, 0].plot(alphas, nf['CL']/nf['CD'], label=airfoil.name)
        ax[1, 1].plot(alphas, nf['Top_Xtr'], linestyle="-", color=color, label=f"{airfoil.name} Top")
        ax[1, 1].plot(alphas, nf['Bot_Xtr'], linestyle="--", color=color, label=f"{airfoil.name} Bot")
        
    ax[0, 0].set_xlabel('CD')
    ax[0, 0].set_ylabel('CL')
    ax[0, 1].set_xlabel('Alpha')
    ax[0, 1].set_ylabel('CL')
    ax[1, 0].set_xlabel('Alpha')
    ax[1, 0].set_ylabel('CL/CD')
    ax[1, 1].set_xlabel('Alpha')
    ax[1, 1].set_ylabel('Xtr')
    for a in ax.flat:
        a.grid(True, linestyle="--", alpha=0.6)
    ax[0,0].legend(loc='center right', )
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig('compare_plot.png')

if __name__ == "__main__":
    """ This program lets you compare airfoils using NeuralFoil. 
    
    There are two structures that are useful here are AirfoilAnalysis and BatchAirfoil.
    BatchAirfoil takes a path to a folder as an input, and analyzes every airfoil given some criteria. 
    BatchAirfoil calls AirfoilAnalysis for each airfoil, which runs NeuralFoil on the airfoil and stores the results.

    TLDR: Use BatchAnalysis to analyze a bunch of airfoils, use AirfoilAnalysis to analyze a single one. 

    BatchAnalysis also has a few helpful functions, such as drawing the analysis, and saving the top performing airfoils.
    """

    """Example Usage:"""
    airfoil_database_path = ".\\coord_seligFmt"
    maxCL_Reynolds = 1.225*10*0.5/(1.7894e-5)
    CL_selection = [0.1, 0.3] # CL values to analyze, can be a single value or a range
    Reynolds = [1239401] # Reynolds numbers to analyze, can also be a range.
    batch = BatchAirfoil(airfoil_database_path, CL_selection, Reynolds, maxCL_Reynolds=maxCL_Reynolds)
    batch.run_batch()
    batch.draw_analysis(save=True,topN=5)
    batch.save_results(topN=20, filename="top20_airfoil.csv")
    batch.save_results(topN=None, filename="full_airfoil.csv")

    """After running the analyses, these were my top performers, 
    and the first airfoil is what I am currently using.
    I want to compare their performance using NeuralFoil, and take a look at their shape."""

    tentative_top = ['sd7032', 'hq2090sm', 'rg12a', 's2048']
    airfoils = [asb.Airfoil(f) for f in tentative_top]
    compare(airfoils=airfoils, reynolds=1239401, save=False)
    for af in airfoils:
        af.draw()


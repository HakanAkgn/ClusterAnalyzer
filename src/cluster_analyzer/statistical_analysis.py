import numpy as np
import pandas as pd
from scipy.special import zeta
from numba import njit, jit
from scipy.stats import linregress
from collections import defaultdict
import powerlaw
import matplotlib.pyplot as plt

plt.style.use('bmh')


def convert_size_counts(row):
    """
    Converts a string representation of size counts into a dictionary.

    Parameters:
        row (str): A string containing size counts in the format 'size1:count1;size2:count2;...'.

    Returns:
        dict: A dictionary with sizes as keys and counts as values.
    """
    if isinstance(row, str):
        try:
            return {float(a.split(":")[0]): float(a.split(":")[1]) for a in row.split(";")}
        except Exception as e:
            print(f"Error processing row: {row} - {e}")
            return {}
    else:
        # Handle cases where row is not a string (e.g., NaN or other unexpected types)
        return {}


def sizes_counts(lambda_df, lmd, TrimLargest=False):
    """
    Aggregates size counts from the dataframe for a specific lambda value.

    Parameters:
        lambda_df (pd.DataFrame): Dataframe containing size counts.
        lmd (float): The lambda value to filter data.
        TrimLargest (bool): If True, trims the largest cluster from the size counts.

    Returns:
        tuple: Arrays of sizes and their corresponding counts.
    """
    all_size_counts = defaultdict(int)
    for size_counts in lambda_df['size_counts']:
        if TrimLargest:
            size_counts_arr = np.array(list(size_counts.items()), dtype=np.uint64)
            if size_counts_arr.size == 0:
                continue  # Skip empty arrays
            if size_counts_arr.ndim == 1:
                size_counts_arr = size_counts_arr.reshape(-1, 2)
            max_cluster_idx = np.argmax(size_counts_arr[:, 0])
            size_counts_arr = np.delete(size_counts_arr, max_cluster_idx, axis=0)
            for size, count in size_counts_arr:
                all_size_counts[size] += count
        else:
            for size, count in size_counts.items():
                all_size_counts[size] += count

    sizes = np.array(list(all_size_counts.keys()), dtype=np.uint64)
    counts = np.array(list(all_size_counts.values()), dtype=np.uint64)
    return sizes, counts


def uniform_downsample(sizes, counts, num_samples):
    """
    Uniformly downsamples the sizes based on counts to a specific number of samples.

    Parameters:
        sizes (np.array): Array of sizes.
        counts (np.array): Array of counts corresponding to sizes.
        num_samples (int): Number of samples to downsample to.

    Returns:
        np.array: Downsampled array of sizes.
    """
    sort_idx = np.argsort(sizes)
    sizes_sorted = sizes[sort_idx]
    counts_sorted = counts[sort_idx]
    counts_bins = counts_sorted.cumsum()
    step = counts_bins[-1] // num_samples
    if step == 0:
        step = 1
    subset_bins = np.arange(0, counts_bins[-1], step)
    subset_idx = np.searchsorted(counts_bins, subset_bins, side='right')
    sizes_samp = sizes_sorted[subset_idx]
    return sizes_samp


@njit
def compute_gof(q, qmins):
    """
    Computes the goodness-of-fit statistics for the power-law distribution.

    Parameters:
        q (np.array): Array of data.
        qmins (np.array): Array of minimum values for the power-law fit.

    Returns:
        list: Goodness-of-fit statistics for each qmin.
    """
    dat = []
    for qmin in qmins:
        zq = q[q >= qmin]
        nq = len(zq)
        a = nq / np.sum(np.log(zq / qmin))
        cq = np.arange(nq) / nq
        cf = 1 - (qmin / zq) ** a
        dat.append(np.max(np.abs(cq - cf)))
    return dat


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_quiet_continuous(x, xmin, reps=1000, limit=None, sample=None, seed=None):
    """
    Performs the continuous method for computing the p-value in a quiet and efficient manner.

    Parameters:
        x (np.array): Data array.
        xmin (float): Minimum value for the power-law behavior.
        reps (int): Number of repetitions for the bootstrap.
        limit (float): Upper limit for qmins.
        sample (int): Number of qmins to sample.
        seed (int): Seed for random number generation.

    Returns:
        tuple: p-value and goodness-of-fit statistic.
    """
    if seed is not None:
        np.random.seed(seed)
    N = len(x)
    nof = []
    z = x[x >= xmin]
    nz = len(z)
    y = x[x < xmin]
    ny = len(y)
    alpha = 1 + nz / np.sum(np.log(z / xmin))
    cz = np.arange(nz) / nz
    cf = 1 - (xmin / np.sort(z)) ** (alpha - 1)
    gof = np.max(np.abs(cz - cf))
    pz = nz / N
    for B in range(reps):
        n1 = np.sum(np.random.random(N) > pz)
        q1 = y[np.random.randint(0, ny, n1)]
        n2 = N - n1
        q2 = xmin * (1 - np.random.random(n2)) ** (-1 / (alpha - 1))
        q = np.sort(np.concatenate((q1, q2)))

        qmins = np.unique(q)[:-1]
        if limit is not None:
            qmins = qmins[qmins <= limit]
        if sample is not None:
            qmins = qmins[np.unique(np.round(np.linspace(0, len(qmins) - 1, sample)).astype(int))]
        dat = compute_gof(q, qmins)
        nof.append(min(dat))
    p = np.sum(np.array(nof) >= gof) / reps
    return p, gof


def plpvalue(
    x,
    xmin,
    vec=np.arange(1.50, 3.51, 0.01),
    reps=1000,
    seed=None,
    quiet=False,
    limit=None,
    sample=None,
    faster_continuous=False,
    force_continuous=False,
    force_discrete=False,
):
    """
    Calculates the p-value for the given power-law fit to some data.

    Parameters:
        x (np.array): Data array.
        xmin (float): Minimum value for the power-law behavior.
        vec (np.array): Array of scaling parameters to consider.
        reps (int): Number of repetitions for the bootstrap.
        seed (int): Seed for random number generation.
        quiet (bool): If True, suppresses output.
        limit (float): Upper limit for qmins.
        sample (int): Number of qmins to sample.
        faster_continuous (bool): If True, uses an optimized continuous method.
        force_continuous (bool): If True, forces the use of the continuous method.
        force_discrete (bool): If True, forces the use of the discrete method.

    Returns:
        tuple: p-value and goodness-of-fit statistic.
    """
    x = np.array(x)
    N = len(x)
    should_continuous = (min(x) > 1000 and len(x) > 100) or np.any(x - np.floor(x) > 0)

    if force_continuous:
        if force_discrete:
            raise ValueError('Cannot force both continuous and discrete methods')
        should_continuous = True
    if force_discrete:
        should_continuous = False

    if should_continuous:
        if not quiet:
            print('Running continuous method')
        if faster_continuous:
            return compute_quiet_continuous(x, xmin, reps=reps, limit=limit, sample=sample, seed=seed)
        else:
            np.random.seed(seed)
            nof = []
            z = x[x >= xmin]
            nz = len(z)
            y = x[x < xmin]
            ny = len(y)
            alpha = 1 + nz / np.sum(np.log(z / xmin))
            cz = np.arange(nz) / nz
            cf = 1 - (xmin / np.sort(z)) ** (alpha - 1)
            gof = np.max(np.abs(cz - cf))
            pz = nz / N

            for B in range(reps):
                n1 = np.sum(np.random.random(N) > pz)
                q1 = y[np.random.randint(0, ny, n1)]
                n2 = N - n1
                q2 = xmin * (1 - np.random.random(n2)) ** (-1 / (alpha - 1))
                q = np.sort(np.concatenate((q1, q2)))

                qmins = np.unique(q)[:-1]
                if limit is not None:
                    qmins = qmins[qmins <= limit]
                if sample is not None:
                    qmins = qmins[
                        np.unique(np.round(np.linspace(0, len(qmins) - 1, sample)).astype(int))
                    ]
                dat = compute_gof(q, qmins)
                nof.append(min(dat))
                if not quiet:
                    print(f'[{B+1}]\tp = {np.sum(np.array(nof) >= gof) / (B+1):.4f}')
            p = np.sum(np.array(nof) >= gof) / reps
            return p, gof

    else:
        if not quiet:
            print('Running discrete method')
        np.random.seed(seed)
        nof = np.array([])
        zvec = zeta(vec, xmin)
        z = x[x >= xmin]
        nz = float(len(z))
        xmax = max(z)
        y = x[x < xmin]
        ny = float(len(y))

        L = -np.inf * np.ones(len(vec))
        slogz = np.sum(np.log(z))
        for k in range(len(vec)):
            L[k] = -nz * np.log(zvec[k]) - vec[k] * slogz
        Y, I = L.max(0), L.argmax(0)
        alpha = vec[I]

        fit = np.cumsum((np.arange(xmin, xmax + 1) ** -alpha) / zvec[I])
        cdi = np.cumsum(
            np.histogram(z, bins=np.arange(xmin, xmax + 2))[0] / nz
        )
        gof = np.max(np.abs(fit - cdi))
        pz = nz / N

        mmax = 20 * xmax
        pdf = np.concatenate(
            [np.zeros(int(xmin) - 1), (np.arange(xmin, mmax + 1) ** -alpha) / zvec[I]]
        )
        cdf = np.vstack(
            [np.arange(1, mmax + 2), np.concatenate([np.cumsum(pdf), [1]])]
        )

        for B in range(reps):
            n1 = np.sum(np.random.random(N) > pz)
            q1 = y[np.random.randint(0, int(ny), n1)]
            n2 = N - n1

            r2 = np.sort(np.random.random(n2))
            c = 0
            q2 = np.zeros(n2, dtype=int)
            k = 0
            for i in range(int(xmin), int(mmax + 2)):
                while c < len(r2) and r2[c] <= cdf[1, i - 1]:
                    c += 1
                q2[k:c] = i
                k = c
                if k >= n2:
                    break
            q = np.concatenate([q1, q2])

            qmins = np.unique(q)
            qmins = qmins[:-1]
            if limit is not None:
                qmins = qmins[qmins <= limit]
            if sample is not None:
                qmins = qmins[
                    np.unique(np.round(np.linspace(0, len(qmins) - 1, sample)).astype(int))
                ]

            dat = np.array([])
            qmax = max(q)
            zq = q
            for qmin in qmins:
                zq = zq[zq >= qmin]
                nq = float(len(zq))
                slogzq = np.sum(np.log(zq))
                if nq > 1:
                    try:
                        L = -nq * np.log(zvec) - vec * slogzq
                    except:
                        L = -np.inf * np.ones(len(vec))
                        for k in range(len(vec)):
                            L[k] = -nq * np.log(zvec[k]) - vec[k] * slogzq
                    Y, I = L.max(0), L.argmax(0)
                    fit = np.cumsum(
                        (np.arange(qmin, qmax + 1) ** -vec[I]) / zvec[I]
                    )
                    cdi = np.cumsum(
                        np.histogram(zq, bins=np.arange(qmin, qmax + 2))[0] / nq
                    )
                    dat = np.append(dat, np.max(np.abs(fit - cdi)))
                else:
                    dat = np.append(dat, -np.inf)

            nof = np.append(nof, np.min(dat))
            if not quiet:
                print(f'[{B + 1}]\tp = {np.sum(nof >= gof) / float(B + 1):.4f}')
        p = np.sum(nof >= gof) / len(nof)
        return p, gof


def perform_plausibility_tests(df, lmds, TrimLargest, downsamples_sizes=[1500]):
    """
    Performs plausibility tests and log-likelihood tests for different lambda values.

    Parameters:
        df (pd.DataFrame): Dataframe containing the data.
        lmds (list): List of lambda values to test.
        TrimLargest (bool): If True, trims the largest cluster from the size counts.
        downsamples_sizes (list): List of downsample sizes.

    Returns:
        None
    """
    col_names = [
        'lambda',
        'alpha',
        's_min',
        'bootstrap-p-u',
        'bootstrap-gof-u',
        'bootstrap-p-r',
        'bootstrap-gof-r',
        'R (pl vs logn)',
        'p (pl vs logn)',
        'R (pl vs strexp)',
        'p (pl vs strexp)',
        'R (pl vs exp)',
        'p (pl vs exp)',
        'R (pl vs pltrc)',
        'p (pl vs pltrc)',
    ]
    rng = np.random.default_rng(42)

    for downsamples_size in downsamples_sizes:
        dat = []
        for lmd in lmds:
            if np.abs(lmd - 0.86) < 0.001:
                continue
            tolerance = 1e-6
            lambda_df = df[np.isclose(df['lambda'], lmd, atol=tolerance)]
            if lambda_df.empty:
                print(f"No data found for lambda = {lmd}")
                continue
            sizes, counts = sizes_counts(lambda_df, lmd, TrimLargest=TrimLargest)
            sizes_samp = uniform_downsample(sizes, counts, 4000000)
            sizes, counts = np.unique(sizes_samp, return_counts=True)

            sorted_idx = np.argsort(sizes)
            sizes_uniq = sizes[sorted_idx]
            counts_uniq = counts[sorted_idx]
            pdf = counts_uniq / np.sum(counts_uniq)

            fit = powerlaw.Fit(sizes_samp, discrete=True, estimate_discrete=True)
            alpha = fit.alpha
            smin = int(fit.xmin)

            sizes_trunc = sizes_uniq[sizes_uniq >= smin]
            counts_trunc = counts_uniq[sizes_uniq >= smin]
            pdf_trunc = counts_trunc / np.sum(counts_trunc)

            downsamples_u = uniform_downsample(sizes_trunc, counts_trunc, downsamples_size)
            downsamples_r = rng.choice(sizes_trunc, downsamples_size, p=pdf_trunc)

            plp_u, plgof_u = plpvalue(
                downsamples_u,
                smin,
                reps=2500,
                seed=42,
                quiet=True, 
                force_continuous=True,
                faster_continuous=True 
            )
            plp_r, plgof_r = plpvalue(
                downsamples_r,
                smin,
                reps=2500,
                seed=2024,
                quiet=True,
                force_continuous=True,
                faster_continuous=True,
            )

            R1, p1 = fit.distribution_compare('power_law', 'lognormal')
            R2, p2 = fit.distribution_compare('power_law', 'stretched_exponential')
            R3, p3 = fit.distribution_compare('power_law', 'exponential')
            R4, p4 = fit.distribution_compare('power_law', 'truncated_power_law', nested=False)
            dat.append(
                [lmd, alpha, smin, plp_u, plgof_u, plp_r, plgof_r, R1, p1, R2, p2, R3, p3, R4, p4]
            )

            print('-' * 75)
            print(f'λ = {lmd:.5f}, α = {alpha:.5f}, smin = {smin}')
            print(
                f'Semi-parametric Bootstrap p-value (uniform downsample) = {plp_u:.5f},  GOF = {plgof_u:.5f}'
            )
            print(
                f'Semi-parametric Bootstrap p-value (random  downsample) = {plp_r:.5f},  GOF = {plgof_r:.5f}'
            )
            print(f'{"R (power_law vs lognormal):":<40s} {R1:<15.5f} p-value: {p1:.5f}')
            print(
                f'{"R (power_law vs stretched_exponential):":<40s} {R2:<15.5f} p-value: {p2:.5f}'
            )
            print(f'{"R (power_law vs exponential):":<40s} {R3:<15.5f} p-value: {p3:.5f}')
            print('\n')
            print(
                f'λ = {lmd:.5f}, α = {alpha:.5f}, smin = {smin}, Size = {downsamples_size}, p_u = {plp_u:.5f}, p_r = {plp_r:.5f}\nR_logn = {R1:.5f}, p_logn = {p1:.5f}, R_strexp = {R2:.5f}, p_strexp = {p2:.5f}, R_exp = {R3:.5f}, p_exp = {p3:.5f}\n'
            )

        dat_df = pd.DataFrame(dat, columns=col_names)
        dat_df.to_csv(f'plvalues_O=10_size={downsamples_size}.csv', index=False)


def generate_plots(df, lmds, TrimLargest=False):
    """
    Generates plots for the power-law analysis.

    Parameters:
        df (pd.DataFrame): Dataframe containing the data.
        lmds (list): List of lambda values to plot.
        TrimLargest (bool): If True, trims the largest cluster from the size counts.

    Returns:
        None
    """
    col_names = [
        'lambda',
        'alpha',
        's_min',
        'R (pl vs logn)',
        'p (pl vs logn)',
        'R (pl vs strexp)',
        'p (pl vs strexp)',
        'R (pl vs exp)',
        'p (pl vs exp)',
        'R (pl vs pltrc)',
        'p (pl vs pltrc)',
    ]
    dat = []

    for lmd in lmds:
        tolerance = 1e-6
        lambda_df = df[np.isclose(df['lambda'], lmd, atol=tolerance)]
        if lambda_df.empty:
            print(f"No data found for lambda = {lmd}")
            continue
        sizes, counts = sizes_counts(lambda_df, lmd, TrimLargest=TrimLargest)

        sorted_idx = np.argsort(sizes)
        sizes_uniq = sizes[sorted_idx]
        counts_uniq = counts[sorted_idx]
        pdf = counts_uniq / np.sum(counts_uniq)
        ccdf = 1 - np.cumsum(pdf)

        sizes_samp = uniform_downsample(sizes, counts, 4000000)

        fit = powerlaw.Fit(sizes_samp, discrete=True, estimate_discrete=True)
        alpha = fit.alpha
        smin = int(fit.xmin)

        sizes_trunc = sizes_uniq[sizes_uniq >= smin]
        counts_trunc = counts_uniq[sizes_uniq >= smin]
        pdf_trunc = counts_trunc / np.sum(counts_trunc)
        ccdf_trunc = 1 - np.cumsum(pdf_trunc)

        R1, p1 = fit.distribution_compare('power_law', 'lognormal')
        R3, p3 = fit.distribution_compare('power_law', 'exponential')
        R2, p2 = fit.distribution_compare('power_law', 'stretched_exponential')
        R4, p4 = fit.distribution_compare('power_law', 'truncated_power_law')

        dat.append([lmd, alpha, smin, R1, p1, R2, p2, R3, p3, R4, p4])

        print(
            f'λ = {lmd:.5f}, α = {alpha:.5f}, smin = {smin}, \nR_logn = {R1:.5f}, p_logn = {p1:.5f}, R_strexp = {R2:.5f}, p_strexp = {p2:.5f}, R_exp = {R3:.5f}, p_exp = {p3:.5f}, R_trunc = {R4:.5f}, p_trunc = {p4:.5f}\n'
        )

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].plot(
            sizes_uniq,
            pdf,
            '.',
            label=f'Empirical PDF\n($λ$={lmd:.5f})',
        )
        ax[0, 0].set(
            xscale='log',
            yscale='log',
            xlabel='Cluster Size (s)',
            ylabel='PDF: $p(S=s)$',
            title=f'Log-Log Plot of Cluster Size Distribution (PDF)',
        )

        ax[0, 1].plot(
            sizes_trunc,
            pdf_trunc,
            '.',
            label=f'Empirical PDF\n($λ$={lmd:.5f})',
        )
        fit.plot_pdf(
            color='k',
            linestyle='--',
            linewidth=2,
            ax=ax[0, 1],
            label=f'Log-binning Histogram',
        )
        fit.power_law.plot_pdf(
            linestyle='-',
            linewidth=2,
            ax=ax[0, 1],
            alpha=0.8,
            label=f'Power-law Fit\n($\\alpha$={alpha:.2f}, $s_{{min}}$={smin:d})',
        )
        ax[0, 1].set(
            xscale='log',
            yscale='log',
            xlabel='Cluster Size (s)',
            ylabel='PDF: $p(S=s)$',
            title=f'Cutoff PDF by KS method',
            ylim=(None, 1.15),
        )

        ax[1, 0].plot(
            sizes_uniq,
            ccdf,
            '.',
            label=f'Empirical cCDF\n($λ$={lmd:.5f})',
        )
        ax[1, 0].set(
            xscale='log',
            yscale='log',
            xlabel='Cluster Size (s)',
            ylabel='cCDF: $Pr(S\\geq s)$',
            title=f'Log-Log Plot of Cluster Size Distribution (cCDF)',
        )

        ax[1, 1].plot(
            sizes_trunc,
            ccdf_trunc,
            '.',
            label=f'Empirical cCDF\n($λ$={lmd:.5f})',
        )
        fit.plot_ccdf(
            color='k',
            linestyle='--',
            linewidth=2,
            ax=ax[1, 1],
            label=f'Log-binning Histogram',
        )
        fit.power_law.plot_ccdf(
            linestyle='-',
            linewidth=2,
            ax=ax[1, 1],
            alpha=0.8,
            label=f'Power-law Fit\n($\\alpha$={alpha:.2f}, $s_{{min}}$={smin:d})',
        )
        ax[1, 1].set(
            xscale='log',
            yscale='log',
            xlabel='Cluster Size (s)',
            ylabel='cCDF: $Pr(S\\geq s)$',
            title=f'Cutoff cCDF by KS method',
            ylim=(None, 1.15),
        )

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()
        plt.tight_layout()
 
    dat_df = pd.DataFrame(dat, columns=col_names)
    dat_df.to_csv(f'plvalues_loglikelihood.csv', index=False)
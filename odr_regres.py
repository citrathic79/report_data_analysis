import pandas
import numpy
import odrpack as odr
import argparse

angle_uncertainty = 1 / 60 / 2  # 1/60 is the precision, so halve it
energy_const = 4.135667696 * 299792458 * 1e-6
rydberg = 10973731.568157
fine_const = 0.0072973525643


def regression_model(measured_data, B):
    theta_initial, wavelen_d1, wavelen_d2 = B[0], B[1], B[2]
    returning = numpy.vstack(
        [
            fun_time(measured_data[0, :], theta_initial, wavelen_d1),
            fun_time(measured_data[1, :], theta_initial, wavelen_d2),
        ]
    )
    return returning


def fun_time(theta_measured, theta_initial, wavelength):
    lines_per_mm = 600
    dist_nm = 1000 * 1000 * 1 / lines_per_mm

    return (
        dist_nm
        / wavelength
        * (
            numpy.sin(numpy.radians(theta_measured + theta_initial))
            - numpy.sin(numpy.radians(theta_initial))
        )
    )


def fit_to_model(group):
    d1_angles = group["D1_Angle"].values.flatten()
    d1_data = d1_angles - d1_angles[2]
    d2_angles = group["D2_Angle"].values.flatten()
    d2_data = d2_angles - d2_angles[2]
    x_data = numpy.vstack((d1_data, d2_data))
    orders = -group["Order"].values
    y_data = numpy.vstack([orders, orders])

    initial_vals = [0, 500, 500]
    bounds = ([-45, 1, 1], [45, 1000, 1000])

    x_error = 1 / (angle_uncertainty**2)
    y_error = 1 / (1e-10**2)

    output = odr.odr_fit(
        f=regression_model,
        xdata=x_data,
        ydata=y_data,
        beta0=initial_vals,
        bounds=bounds,
        weight_x=x_error,
        weight_y=y_error,
    )
    p_opt = output.beta
    p_se = output.sd_beta

    return pandas.Series(
        {
            "incident angle": p_opt[0],
            "incident angle err": p_se[0],
            "wavelength d1": p_opt[1],
            "wavelength d1 err": p_se[1],
            "wavelength d2": p_opt[2],
            "wavelength d2 err": p_se[2],
        }
    )


def effective_charge(energy):
    return (54 * energy / (rydberg * (fine_const**2) * energy_const / 1e9)) ** (1 / 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    df = pandas.read_csv(args.path)
    print(df)

    results = df.groupby("Trial", as_index=True).apply(fit_to_model)
    print(results)

    results.to_csv("wavelength_odr.csv")

    agg = pandas.Series()
    agg["lambda_d1"] = results["wavelength d1"].sum() / 3
    agg["lambda_d1 err"] = numpy.sqrt((results["wavelength d1 err"] ** 2).sum() / 3)
    agg["lambda_d2"] = results["wavelength d2"].sum() / 3
    agg["lambda_d2 err"] = numpy.sqrt((results["wavelength d2 err"] ** 2).sum() / 3)

    agg["E_d1"] = energy_const / agg["lambda_d1"]
    agg["E_d1 err"] = agg["lambda_d1 err"] / agg["lambda_d1"] * agg["E_d1"]
    agg["E_d2"] = energy_const / agg["lambda_d2"]
    agg["E_d2 err"] = agg["lambda_d2 err"] / agg["lambda_d2"] * agg["E_d2"]
    agg["deltaE"] = abs(agg["E_d1"] - agg["E_d2"])
    agg["deltaE err"] = agg["E_d1 err"] + agg["E_d2 err"]
    agg["Z_0"] = effective_charge(agg["deltaE"])
    agg["Z_0 err"] = (
        effective_charge(agg["deltaE"] + agg["deltaE err"])
        - effective_charge(agg["deltaE"] - agg["deltaE err"])
    ) / 2

    print(agg)
    agg.to_csv("aggregated_data.csv")

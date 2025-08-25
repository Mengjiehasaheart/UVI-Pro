
# Minute UV Engine from Hourly UV Index to Minute resolved for Erythemal Irradiance and Dose
## Mengjie Fan, 2025 Summer
## Overview

This application presents a method, validation protocol, and reference implementation for transforming current hourly forecasts of the ultraviolet index (UVI, commonly seen in modern day weather app) into minute resolved fields of UVI, erythemally weighted irradiance $E_{\mathrm{ery}}$ in $W\,m^{-2}$, and cumulative erythemal dose in $J\,m^{-2}$ and Standard Erythema Dose (SED). The downscaling is physics aware yet lightweight with sub hour variability follows the solar zenith geometry through a normalized $\mu^p$ shape with $\mu=\cos\theta$ whilst cloud effects are incorporated by a cloud modification factor (CMF) constructed from the ratio of total sky to clear sky hourly UVI and interpolated in time. The idea is to preserve the provider’s hourly means by construction along with dimensional consistency at every step accompanied by produces diagnostic reaggregation plots and error metrics. An accompanying command line application fetches live hourly UVI and clear sky UVI for any coordinate and date via Open‑Meteo and generates both numerical outputs and visualizations. A static example for Sicily island is included.

As known by many humans already, public health agencies encourage the use of the ultraviolet index to communicate the risk of erythemal skin damage under solar radiation exposure, tho usually not an concern for British humans. Operational forecast systems typically disseminate UVI at hourly cadence, which is adequate for awareness but coarse for exposure modeling, wearables, or irrational behavior linked advice. The present work addresses the problem of deriving minute scale UVI and erythemal quantities from hourly inputs without resorting to a full radiative transfer model. The design goal is to remain faithful to basic radiative geometry and to the separation, common in modern systems, between clear sky and total sky (cloud‑affected) conditions.

Typically, the erythemally weighted irradiance $E_{\mathrm{ery}}(t)$ is defined using the CIE erythema reference action spectrum, which weights ultraviolet wavelengths according to their effectiveness for producing erythema. By international human convention, the ultraviolet index is proportional to $E_{\mathrm{ery}}$ through
$$
\mathrm{UVI}(t)=40\,E_{\mathrm{ery}}(t),
$$
when $E_{\mathrm{ery}}$ is expressed in $W\,m^{-2}$. The transformation used in the software therefore reads
$$
E_{\mathrm{ery}}(t)=\frac{\mathrm{UVI}(t)}{40}\quad [W·m⁻²].
$$
The cumulative erythemal dose, also called erythemal radiant exposure, is the time integral of $E_{\mathrm{ery}}$,
$$
H_{\mathrm{ery}}(t)=\int_{t_0}^{t} E_{\mathrm{ery}}(\tau)\,\mathrm{d}\tau\quad [J·m⁻²].
$$
The Standard Erythema Dose is defined by
$$
\mathrm{SED}=\frac{H_{\mathrm{ery}}}{100},
$$
so that $1\,\mathrm{SED} = 100\,J\,m^{-2}$. In the minute resolved implementation which we desired with a fixed time step $\Delta t=60\,\mathrm{s}$, these relations appear in discrete form as
$$
H_{\mathrm{ery}}(t_n)\approx \sum_{k=1}^{n} E_{\mathrm{ery}}(t_k)\,\Delta t,\qquad
\mathrm{SED}(t_n)=\frac{1}{100}\sum_{k=1}^{n} E_{\mathrm{ery}}(t_k)\,60.
$$

## Data and Notation

Two hourly fields are required at the target location and date, being the total‑sky UVI, denoted $\mathrm{UVI}^{\mathrm{all}}_h$, and the clear sky UVI, denoted $\mathrm{UVI}^{\mathrm{clr}}_h$, both time stamped at the beginning of hour $h$ in local civil time. The reference implementation retrieves these from Open‑Meteo, which exposes `uv_index` and `uv_index_clear_sky` in its hourly endpoint and provides the appropriate IANA timezone string for the request. The location is specified by latitude $\phi$ and longitude $\lambda$ in degrees. In my application, angles in trigonometric functions are in radians unless otherwise noted, irradiances are in $W\,m^{-2}$, doses in $J\,m^{-2}$, and UVI is dimensionless.

Firstly, minute scale changes in clear sky irradiance are dominated by the variation of the solar zenith angle $\theta(t)$. The cosine of the zenith angle, $\mu(t)=\cos\theta(t)$, is computed using standard solar position formulae. Let $N$ be the day of year and $h_{\mathrm{LST}}$ the local solar time in hours. The fractional year is
$$
\Gamma = \frac{2\pi}{365}\left(N-1+\frac{h_{\mathrm{LST}}-12}{24}\right).
$$
The equation of time $\mathrm{EOT}$ in minutes and the solar declination $\delta$ in radians follow the NOAA/Spencer trigonometric expansions,
$$
\mathrm{EOT}=229.18\left(0.000075+0.001868\cos\Gamma-0.032077\sin\Gamma-0.014615\cos 2\Gamma-0.040849\sin 2\Gamma\right),
$$
$$
\delta = 0.006918-0.399912\cos\Gamma+0.070257\sin\Gamma-0.006758\cos 2\Gamma+0.000907\sin 2\Gamma-0.002697\cos 3\Gamma+0.00148\sin 3\Gamma.
$$
With longitude $\lambda$ in degrees east and timezone offset $\mathrm{TZ}_{\mathrm{offset}}$ in minutes, the true solar time in minutes is
$$
\mathrm{TST}=\left(m_{\mathrm{clock}}+\mathrm{EOT}+4\lambda-\mathrm{TZ}_{\mathrm{offset}}\right)\bmod 1440,
$$
and the hour angle $\mathrm{HA}$ in radians is
$$
\mathrm{HA}=\frac{\pi}{180}\left(\frac{\mathrm{TST}}{4}-180\right).
$$
The zenith cosine then reads
$$
\mu(t)=\sin\phi\,\sin\delta+\cos\phi\,\cos\delta\,\cos(\mathrm{HA}).
$$
Although my algorithm does not require an explicit optical airmass, still for reference we compute the Kasten–Young (1989) approximation for $\theta<90^\circ$,
$$
m(\theta)=\bigl[\cos\theta+0.50572\,(96.07995^\circ-\theta)^{-1.6364}\bigr]^{-1},
$$
note this is a dimensionless quantity widely used in solar resource assessment.

## Downscaling Algorithm

The hour $h$ is associated with the set $\mathcal{T}_h$ of minute timestamps within $[h,h+1)$. The clear‑sky minute field is constructed by distributing the hourly clear‑sky UVI in proportion to $\mu(t)^p$ over $\mathcal{T}_h$, where $p$ is a tunable exponent typically close to unity. Defining the unnormalized shape $S_h^{\star}(t)=\mu(t)^p$ for $t\in\mathcal{T}_h$, the scale factor $\alpha_h$ is chosen to preserve the provider’s hourly mean exactly,
$$
\alpha_h=\frac{\mathrm{UVI}^{\mathrm{clr}}_h\,|\mathcal{T}_h|}{\sum_{t\in\mathcal{T}_h} S_h^{\star}(t)},
\qquad
\mathrm{UVI}^{\mathrm{clr}}(t)=\alpha_h\,S_h^{\star}(t)\quad\text{for}\ t\in\mathcal{T}_h.
$$
Cloud effects are introduced through the cloud modification factor. At hourly resolution the CMF is
$$
\mathrm{CMF}_h=\frac{\mathrm{UVI}^{\mathrm{all}}_h}{\mathrm{UVI}^{\mathrm{clr}}_h},
$$
which is interpolated to a continuous time function $\mathrm{CMF}(t)$ and then sampled at minute cadence. In the reference code, temporal interpolation is performed in the provider’s local time; values are constrained to a physically plausible interval such as $[0,1.3]$ to permit mild cloud edge enhancements while preventing numerical outliers. The total sky minute UVI is obtained by
$$
\mathrm{UVI}^{\mathrm{all}}(t)=\mathrm{UVI}^{\mathrm{clr}}(t)\,\mathrm{CMF}(t).
$$
Erythemally weighted irradiance and dose follow from the definitions in the preceding section through a mere change of units and a cumulative sum.

## Validation

Internal consistency is assessed by reaggregating the minute results to hourly means and comparing them with the provider’s hourly UVI. Denoting by $\widehat{\mathrm{UVI}}^{\mathrm{all}}_h$ the mean of $\mathrm{UVI}^{\mathrm{all}}(t)$ over $\mathcal{T}_h$, the error statistics reported are the mean absolute error and the RMSE,
$$
\mathrm{MAE}=\frac{1}{H}\sum_{h=1}^{H}\left|\widehat{\mathrm{UVI}}^{\mathrm{all}}_h-\mathrm{UVI}^{\mathrm{all}}_h\right|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{H}\sum_{h=1}^{H}\left(\widehat{\mathrm{UVI}}^{\mathrm{all}}_h-\mathrm{UVI}^{\mathrm{all}}_h\right)^2},
$$
with $H$ the number of valid hours. Two diagnostic figures accompany these metrics: a one to one scatter between $\widehat{\mathrm{UVI}}^{\mathrm{all}}_h$ and $\mathrm{UVI}^{\mathrm{all}}_h$, and the error series as a function of time. Because the clear sky hourly means are preserved identically by construction, the only sources of discrepancy are the temporal interpolation of $\mathrm{CMF}(t)$ and any provider specific phase differences in the hourly time stamps.

The method is implemented in a program `uv_minute_app.py`. The application fetches hourly `uv_index` and `uv_index_clear_sky` for a user specified latitude, longitude, and date, and subsequently constructs a timezone aware minute grid covering the day 

## Example for Sicily

A static example was made for Sicily island this summer 2025 whilst I was there to illustrate the workflow and the expected structure of outputs. The example focuses on Palermo and reproduces the principal figures as the minute resolved UVI under clear and total sky conditions and the cumulative erythemal dose together with the two validation plots. 

Last thing to keep in mind that the preservation of hourly clear sky means constrains one component of the problem exactly but the temporal structure of cloud attenuation below the hourly scale remains an approximation when derived from interpolation of the hourly CMF. For applications that require sharper timing, the present framework can be extended by constructing $\mathrm{CMF}(t)$ directly from satellite cloud nowcasts while retaining the same clear sky minute construction. The method is agnostic to the backend supplying hourly UVI and can be adapted to services that provide erythemally weighted dose rate instead, and also conversion to UVI requires only the proportionality $ \mathrm{UVI}=40\,E_{\mathrm{ery}} $ with $E_{\mathrm{ery}}$ in $W\,m^{-2}$. The dose definitions and units used here follow the CIE standard and remain valid irrespective of the data provider.

## Usage

The following example invokes the program for Catania in Sicily on 2025‑08‑17 using the Europe/Rome timezone and writes all outputs to a directory named `outputs`.
```
python uv_minute_app.py --lat 37.5079 --lon 15.0830 --date 2025-08-17 --timezone Europe/Rome --outdir outputs --html
```
It also prints a small JSON report to standard output and populates the output directory with the CSV, figures, and validation metrics. All timestamps in the CSV are expressed in the local timezone reported by the data provider.


## References

Commission Internationale de l’Éclairage (CIE). Erythema Reference Action Spectrum and Standard Erythema Dose. CIE publication.

Kasten, F., and A. T. Young, 1989. Revised optical air mass tables and approximation formula. Applied Optics, 28(22), 4735–4738.

Open‑Meteo. Weather Forecast API documentation describing the `uv_index` and `uv_index_clear_sky` hourly variables and timezone handling.

World Health Organization and World Meteorological Organization. Global Solar UV Index: A Practical Guide. The defining proportionality $\mathrm{UVI}=40\,E_{\mathrm{ery}}$ with $E_{\mathrm{ery}}$ in $W\,m^{-2}$ is adopted for public communication.

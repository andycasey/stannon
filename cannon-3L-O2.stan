
// The Stannon

// A 3-label, second-order (O2; quadratic) model

data {
    int<lower=1> P; // number of pixels
    int<lower=1> S; // number of training set stars
    int<lower=1> L; // number of training set labels
    
    matrix[S, P] y; // pseudo-continuum-normalized flux
    matrix[S, P] y_var; // the variance on the flux values y

    matrix[S, L] label; // the training set labels (central values)
    matrix[S, L] label_err; // the error in the training set label
                             
    real f; // divide the mean-offset labels by f\sigma to be ~isotropic
}

transformed data {
    matrix[S, L] iso_label; // ~isotropic-scaled labels

    // Construct the iso_labels matrix
    for (l in 1:L) {
        real central;
        real sigma;

        central = mean(label[:, l]);
        sigma = sd(label[:, l]);
        for (s in 1:S)
            iso_label[s, l] = (label[s, l] - central)/(f * sigma);
    }
}

parameters {
    vector[P] eps_variance; // the variance in each pixel
    // 10 is the number of terms for a quadratic model
    matrix[P, 10] theta; // the spectral derivatives
    matrix[S, L] true_label; // the true label values 
}

transformed parameters {
    // 10 is the number of terms for a quadratic model
    matrix[S, 10] DM; // design matrix
    matrix[S, P] y_err; // total uncertainties

    // Construct the design matrix from the labels
    DM = rep_matrix(1.0, S, 10);
    for (s in 1:S) {
        DM[s, 2] = iso_label[s, 1]; // teff
        DM[s, 3] = iso_label[s, 2]; // logg
        DM[s, 4] = iso_label[s, 3]; // feh
        DM[s, 5] = pow(iso_label[s, 1], 2); // teff^2
        DM[s, 6] = pow(iso_label[s, 2], 2); // logg^2
        DM[s, 7] = pow(iso_label[s, 3], 2); // feh^2
        DM[s, 8] = iso_label[s, 1] * iso_label[s, 2]; // teff * logg
        DM[s, 9] = iso_label[s, 1] * iso_label[s, 3]; // teff * feh
        DM[s, 10] = iso_label[s, 2] * iso_label[s, 3]; // logg * feh
    }
    for (s in 1:S)
        y_err[s] = sqrt(to_vector(y_var[s]) + eps_variance)';
}

model {
    //eps_variance ~ normal(0, 0.01); // prior that scatter term is probably small.

    // priors on the labels
    for (l in 1:L)
        label[:, l] ~ normal(true_label[:, l], label_err[:, l]);

    // model the fluxes.
    for (s in 1:S) {
        y[s] ~ normal(theta * DM[s]', y_err[s]);
    }
}
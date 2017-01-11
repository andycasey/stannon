
// The Stannon

data {
    int<lower=1> P; // number of pixels
    int<lower=1> S; // number of training set stars
    int<lower=1> L; // number of training set labels
    int<lower=1> O; // maximum order of the polynomial model (O = 2 --> T^2 + ...)

    matrix[S, P] y; // pseudo-continuum-normalized flux
    matrix[S, P] y_err; // 1\sigma errors on flux y

    matrix[S, L] labels; // the training set labels (central values)
    matrix[S, L] label_err; // the error in the training set label
                             
    real f; // divide the mean-offset labels by f\sigma to be ~isotropic
}

transformed data {
    int C; // number of theta coefficients per pixel
    matrix[S, L] iso_labels; // ~isotropic-scaled labels

    // The indices NI, LI, and OI are to construct the vectorizer
    int NI[10]; // the number of terms in each design matrix entry
    int LI[10, O]; // label indices for each design matrix entry
    int OI[10, O]; // order indices for each design matrix entry

    //C = num_coefficients(O);
    // This asserts L = 3 and O = 2
    C = 10;
    NI[1] = 1;
    LI[1, 1] = 1;
    OI[1, 1] = 1;
    NI[2] = 1;
    LI[2, 1] = 2;
    OI[2, 1] = 1;
    NI[3] = 1;
    LI[3, 1] = 3;
    OI[3, 1] = 1;
    NI[4] = 1;
    LI[4, 1] = 1;
    OI[4, 1] = 2;
    NI[5] = 2;
    LI[5, 1] = 1;
    OI[5, 1] = 1;
    LI[5, 2] = 2;
    OI[5, 2] = 1;
    NI[6] = 2;
    LI[6, 1] = 1;
    OI[6, 1] = 1;
    LI[6, 2] = 3;
    OI[6, 2] = 1;
    NI[7] = 1;
    LI[7, 1] = 2;
    OI[7, 1] = 2;
    NI[8] = 2;
    LI[8, 1] = 2;
    OI[8, 1] = 1;
    LI[8, 2] = 3;
    OI[8, 2] = 1;
    NI[9] = 1;
    LI[9, 1] = 3;
    OI[9, 1] = 2;

    // PYTHON CODE THAT FILLS IN N1, L1, O1 //

    // Construct the iso-labels matrix
    for (l in 1:L) {
        real central;
        real sigma;

        central = mean(labels[:, l]);
        sigma = sd(labels[:, l]);
        for (s in 1:S)
            iso_labels[s, l] = (labels[s, l] - central)/(f * sigma);
    }
}

parameters {
    vector[P] eps; // the noise in each pixel
    matrix[P, C] theta; // the spectral derivatives
    matrix[S, L] true_labels; // the true label values 
}

transformed parameters {
    matrix[S, C] DM; // design matrix

    // Construct the design matrix from the labels
    {
        DM = rep_matrix(1.0, S, C);

        for (c in 2:C) { // 2 because we filled the first entry.
            int K;
            K = NI[c];
            {
                for (n in 1:K) {
                    int li;
                    int oi;

                    li = LI[c, n];
                    oi = OI[c, n];

                    for (s in 1:S)
                        DM[s, c] = DM[s, c] * pow(iso_labels[s, li], oi);
                }
            }
        }
    }
}

model {
    eps ~ normal(0, 0.01);
    for (l in 1:L)
        labels[:, l] ~ normal(true_labels[:, l], label_err[:, l]);
    for (s in 1:S)
        y[s] ~ normal(theta * DM[s]', to_vector(y_err[s]) + eps);
}
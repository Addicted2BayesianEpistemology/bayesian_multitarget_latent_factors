

data {

  // dimensions
  int<lower=1> N;             // Number of Samples
  
  int<lower=1> L1;             // Number of Locations of Observation for branch 1
  int<lower=1> L2;             // Number of Locations of Observation for branch 2
  
  int<lower=1> p1;             // Number of basis functions for branch 1
  int<lower=1> p2;             // Number of basis functions for branch 2
  
  int<lower=1> k;             // Number of Latent Factors
  int<lower=1> r;             // Number of Covariates

  // data
  matrix[L1,N] y1;             // target for branch 1
  matrix[L2,N] y2;             // target for branch 2
  matrix[r,N] X;             // covariates

  // basis functions matrices
  matrix[L1,p1] B1;
  matrix[L2,p2] B2;
  
  // hyperparameters
  real<lower=0> v;           // degrees of freedom for the student_t distribution of the latent factors (Lambda) prior
  real<lower=0> nu;          // degrees of freedom for the student_t distribution of the regression coefficients (beta) prior
  real<lower=0> alpha_psi;   // alpha hyperparameter inverse gamma prior for the noise of y about f
  real<lower=0> beta_psi;    // beta hyperparameter inverse gamma prior for the noise of y about f
  real<lower=0> psi_noise_scale_multiplier_1;
  real<lower=0> psi_noise_scale_multiplier_2;
  real<lower=0> alpha_sigma; // alpha hyperparameter gamma prior for the precision of basis function coefficients
  real<lower=0> beta_sigma;  // beta hyperparameter gamma prior for the precision of basis function coefficients
  real<lower=0> theta_noise_scale_multiplier_1;
  real<lower=0> theta_noise_scale_multiplier_2;
  real<lower=0> alpha_1;     // alpha hyperparameter gamma prior for delta_1
  real<lower=0> alpha_2;     // alpha hyperparameter gamma prior for delta_2:k

}


transformed data {

  matrix[L1+L2,N] y_all;
  y_all[1:L1,:] = y1;
  y_all[(L1+1):(L1+L2),:] = y2;
  
}


parameters {
  matrix[p1,N] theta1;
  matrix[p2,N] theta2;
  matrix[p1,k] Lambda1;
  matrix[p2,k] Lambda2;
  matrix[k,N] eta;
  matrix[k,r] beta;
  vector<lower=0>[p1] tau_theta1;
  vector<lower=0>[p2] tau_theta2;
  vector<lower=0>[k] delta1;
  vector<lower=0>[k] delta2;
  real<lower=0> psi1;
  real<lower=0> psi2;
}


transformed parameters {
  vector[k] tau_lambda1;
  tau_lambda1[1] = delta1[1];
  for (h in 2:k) {
    tau_lambda1[h] = prod(delta1[1:h]);
  }
  vector[k] tau_lambda2;
  tau_lambda2[1] = delta2[1];
  for (h in 2:k) {
    tau_lambda2[h] = prod(delta2[1:h]);
  }
}


model {

  // lklhood
  to_vector(y1 - B1*theta1) ~ normal(0, psi1*psi_noise_scale_multiplier_1);
  to_vector(y2 - B2*theta2) ~ normal(0, psi2*psi_noise_scale_multiplier_2);
  to_vector(diag_pre_multiply(sqrt(tau_theta1), theta1 - Lambda1*eta)) ~ normal(0,theta_noise_scale_multiplier_1);
  to_vector(diag_pre_multiply(sqrt(tau_theta2), theta2 - Lambda2*eta)) ~ normal(0,theta_noise_scale_multiplier_2);
  to_vector(eta - beta*X) ~ normal(0,1);

  // priors for matrices
  to_vector(diag_post_multiply(Lambda1, sqrt(tau_lambda1))) ~ student_t(v, 0, 1);
  to_vector(diag_post_multiply(Lambda2, sqrt(tau_lambda2))) ~ student_t(v, 0, 1);
  //  to_vector(beta) ~ cauchy(0,1);
  to_vector(beta) ~ student_t(nu, 0,1);

  // priors
  square(psi1) ~ inv_gamma(alpha_psi, beta_psi);
  square(psi2) ~ inv_gamma(alpha_psi, beta_psi);
  tau_theta1 ~ gamma(alpha_sigma, beta_sigma);
  tau_theta2 ~ gamma(alpha_sigma, beta_sigma);
  delta1[1] ~ gamma(alpha_1, 1);
  delta1[2:k] ~ gamma(alpha_2, 1);
  delta2[1] ~ gamma(alpha_1, 1);
  delta2[2:k] ~ gamma(alpha_2, 1);

  // Jacobian Corrections
  target += log(psi1);
  target += log(psi2);
  target += N*0.5*sum(log(tau_theta1));
  target += p1*0.5*sum(log(tau_lambda1));
  target += N*0.5*sum(log(tau_theta2));
  target += p2*0.5*sum(log(tau_lambda2));
}


generated quantities {

  matrix[L1,r] regr_coeffs1 = B1*Lambda1*beta;
  matrix[L1,N] estimated_y1 = regr_coeffs1*X;
  matrix[L2,r] regr_coeffs2 = B2*Lambda2*beta;
  matrix[L2,N] estimated_y2 = regr_coeffs2*X;

  matrix[L1,L1] Sigma_11 = add_diag( B1*( add_diag( Lambda1*(Lambda1') , square(theta_noise_scale_multiplier_1)./tau_theta1 ) )*(B1') , square(psi1*psi_noise_scale_multiplier_1) );
  matrix[L2,L2] Sigma_22 = add_diag( B2*( add_diag( Lambda2*(Lambda2') , square(theta_noise_scale_multiplier_2)./tau_theta2 ) )*(B2') , square(psi2*psi_noise_scale_multiplier_2) );
  matrix[L1,L2] Sigma_12 = B1*Lambda1*(Lambda2')*(B2');
  matrix[L2,L1] Sigma_21 = B2*Lambda2*(Lambda1')*(B1');
  
  array[N] real log_lik_y;
  matrix[L1,N] y1_predictive;
  matrix[L2,N] y2_predictive;
  {
    matrix[L1+L2,L1+L2] Sigma;
    Sigma[1:L1,1:L1] = Sigma_11;
    Sigma[1:L1,(L1+1):(L1+L2)] = Sigma_12;
    Sigma[(L1+1):(L1+L2),1:L1] = Sigma_21;
    Sigma[(L1+1):(L1+L2),(L1+1):(L1+L2)] = Sigma_22;

    matrix[L1+L2,L1+L2] Choleksy_Decomposed_Sigma = cholesky_decompose(Sigma);
    
    for (i in 1:N) {
      vector[L1+L2] estimated_y_aux;
      estimated_y_aux[1:L1] = estimated_y1[:,i];
      estimated_y_aux[(L1+1):(L1+L2)] = estimated_y2[:,i];
      
      vector[L1+L2] y_predictive_aux = multi_normal_cholesky_rng(estimated_y_aux, Choleksy_Decomposed_Sigma);
      log_lik_y[i] = multi_normal_cholesky_lpdf(y_all[:,i] | estimated_y_aux, Choleksy_Decomposed_Sigma);
      
      y1_predictive[:,i] = y_predictive_aux[1:L1];
      y2_predictive[:,i] = y_predictive_aux[(L1+1):(L1+L2)];
    }
  }
  
}

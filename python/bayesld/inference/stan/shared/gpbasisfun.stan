// Implemented from 
// https://users.aalto.fi/~ave/casestudies/Motorcycle/motorcycle.html
vector diagSPD_EQ(real alpha, real rho, real L, int M) {
  return alpha * sqrt(sqrt(2 * pi()) * rho)
         * exp(-0.25 * (rho * pi() / 2 / L)^2 * linspaced_vector(M, 1, M)^2);
}
matrix PHI(int N, int M, real L, vector x) {
  return sin(diag_post_multiply(rep_matrix(pi() / (2 * L) * (x + L), M),
                                linspaced_vector(M, 1, M))) / sqrt(L);
}

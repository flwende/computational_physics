#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>

// bits
#define PRECISION 512

static double J;
static double K;
static int M;
static int N;

static double FF_gamma_r(const int r)
{
  double
    returnValue,
    c_l;

  if(r == 0) {
    returnValue = 2.0*K+log(tanh(K));
  } else {
    c_l = (cosh(2.0*K)/tanh(2.0*K))-cos(r*M_PI/N);
    returnValue = log(c_l+sqrt(c_l*c_l-1.0));
  }

  return (double) returnValue;
}

static double FF_gamma_r_prime(const int r)
{
  double
    returnValue,
    c_l,
    c_l_prime;

  if(r == 0) {
    returnValue = 2.0*(1.0+1.0/sinh(2.0*K));
  } else {
    c_l = (cosh(2.0*K)/tanh(2.0*K))-cos(r*M_PI/N);
    c_l_prime = 2.0*cosh(2.0*K)*(1.0-1.0/(sinh(2.0*K)*sinh(2.0*K)));
    returnValue = c_l_prime/sqrt(c_l*c_l-1.0);
  }

  return (double) returnValue;
}

static double FF_gamma_r_pprime(const int r)
{
  double
    returnValue,
    c_l,
    c_l_prime,
    c_l_pprime;
  
  if(r == 0) {
    returnValue = -4.0/(sinh(2.0*K)*tanh(2.0*K));
  } else {
    c_l = (cosh(2.0*K)/tanh(2.0*K))-cos(r*M_PI/N);
    c_l_prime = 2.0*cosh(2.0*K)*(1.0-1.0/(sinh(2.0*K)*sinh(2.0*K)));
    c_l_pprime  = 8.0/(sinh(2.0*K)*sinh(2.0*K)*sinh(2.0*K))*cosh(2.0*K)*cosh(2.0*K);
    c_l_pprime += 4.0*(sinh(2.0*K)-1.0/sinh(2.0*K));

    returnValue  = c_l_pprime/sqrt(c_l*c_l-1.0);
    returnValue -= c_l_prime*c_l_prime*c_l/(sqrt(c_l*c_l-1.0)*sqrt(c_l*c_l-1.0)*sqrt(c_l*c_l-1.0));
  }

  return (double) returnValue;
}

static void FF_partitionFunction_Zx(mpfr_t *result, const int x)
{
  int
    i;
  long double
    ld_temp;
  mpfr_t
    mpfr_temp,
    mpfr_two;

  mpfr_init2(mpfr_temp,PRECISION);
  mpfr_init2(mpfr_two,PRECISION);

  mpfr_set_ld((*result),1.0,GMP_RNDN);
  mpfr_set_ld(mpfr_two,2.0,GMP_RNDN);

  if(x == 1) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp,ld_temp,GMP_RNDN);
      mpfr_cosh(mpfr_temp,mpfr_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp,mpfr_temp,mpfr_two,GMP_RNDN);
      mpfr_mul((*result),(*result),mpfr_temp,GMP_RNDN);
    }
  } else if(x == 2) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp,ld_temp,GMP_RNDN);
      mpfr_sinh(mpfr_temp,mpfr_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp,mpfr_temp,mpfr_two,GMP_RNDN);
      mpfr_mul((*result),(*result),mpfr_temp,GMP_RNDN);
    }
  } else if(x == 3) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp,ld_temp,GMP_RNDN);
      mpfr_cosh(mpfr_temp,mpfr_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp,mpfr_temp,mpfr_two,GMP_RNDN);
      mpfr_mul((*result),(*result),mpfr_temp,GMP_RNDN);
    }
  } else if(x == 4) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp,ld_temp,GMP_RNDN);
      mpfr_sinh(mpfr_temp,mpfr_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp,mpfr_temp,mpfr_two,GMP_RNDN);
      mpfr_mul((*result),(*result),mpfr_temp,GMP_RNDN);
    }
  }
  
  mpfr_clear(mpfr_temp);
  mpfr_clear(mpfr_two);
}

static void FF_partitionFunction_ZxPrime_div_Zx(mpfr_t *result, const int x) 
{
  int
    i;
  long double
    ld_temp;
  mpfr_t
    mpfr_temp1,
    mpfr_temp2;

  mpfr_init2(mpfr_temp1,PRECISION);
  mpfr_init2(mpfr_temp2,PRECISION);
  mpfr_set_ld((*result),0.0,GMP_RNDN);
  
  if(x == 1) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp1,mpfr_temp1,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i+1);
      mpfr_set_ld(mpfr_temp2,ld_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp1,mpfr_temp2,mpfr_temp1,GMP_RNDN);
      mpfr_add((*result),(*result),mpfr_temp1,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
    mpfr_mul((*result),(*result),mpfr_temp1,GMP_RNDN);
  } else if(x == 2) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp1,mpfr_temp1,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i+1);
      mpfr_set_ld(mpfr_temp2,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp1,mpfr_temp2,mpfr_temp1,GMP_RNDN);
      mpfr_add((*result),(*result),mpfr_temp1,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
    mpfr_mul((*result),(*result),mpfr_temp1,GMP_RNDN);
  } else if(x == 3) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp1,mpfr_temp1,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i);
      mpfr_set_ld(mpfr_temp2,ld_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp1,mpfr_temp2,mpfr_temp1,GMP_RNDN);
      mpfr_add((*result),(*result),mpfr_temp1,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
    mpfr_mul((*result),(*result),mpfr_temp1,GMP_RNDN);
  } else if(x == 4) {
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp1,mpfr_temp1,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i);
      mpfr_set_ld(mpfr_temp2,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp1,mpfr_temp2,mpfr_temp1,GMP_RNDN);
      mpfr_add((*result),(*result),mpfr_temp1,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp1,ld_temp,GMP_RNDN);
    mpfr_mul((*result),(*result),mpfr_temp1,GMP_RNDN);
  }

  mpfr_clear(mpfr_temp1);
  mpfr_clear(mpfr_temp2);
}

static void FF_partitionFunction_ZxPPrime_div_Zx(mpfr_t *result, const int x) 
{
  int
    i;
  long double
    ld_temp;
  mpfr_t
    mpfr_temp1,
    mpfr_temp2,
    mpfr_temp3,
    mpfr_temp4;

  mpfr_init2(mpfr_temp1,PRECISION);
  mpfr_init2(mpfr_temp2,PRECISION);
  mpfr_init2(mpfr_temp3,PRECISION);
  mpfr_init2(mpfr_temp4,PRECISION);

  if(x == 1) {
    mpfr_set_ld(mpfr_temp1,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i+1);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp1,GMP_RNDN);

    mpfr_set_ld(mpfr_temp2,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_pprime(2*i+1);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);

      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_cosh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) (0.5*M*FF_gamma_r_prime(2*i+1)*FF_gamma_r_prime(2*i+1));
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    
    mpfr_add((*result),mpfr_temp1,mpfr_temp2,GMP_RNDN);
  } else if(x == 2) {
    mpfr_set_ld(mpfr_temp1,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i+1);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp1,GMP_RNDN);

    mpfr_set_ld(mpfr_temp2,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_pprime(2*i+1);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);

      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i+1));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_sinh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) (0.5*M*FF_gamma_r_prime(2*i+1)*FF_gamma_r_prime(2*i+1));
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_sub(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    
    mpfr_add((*result),mpfr_temp1,mpfr_temp2,GMP_RNDN);
  } else if(x == 3) {
    mpfr_set_ld(mpfr_temp1,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp1,GMP_RNDN);

    mpfr_set_ld(mpfr_temp2,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_pprime(2*i);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);

      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_cosh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) (0.5*M*FF_gamma_r_prime(2*i)*FF_gamma_r_prime(2*i));
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    
    mpfr_add((*result),mpfr_temp1,mpfr_temp2,GMP_RNDN);
  } else if(x == 4) {
    mpfr_set_ld(mpfr_temp1,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_prime(2*i);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp3,GMP_RNDN);
    mpfr_mul(mpfr_temp1,mpfr_temp1,mpfr_temp1,GMP_RNDN);

    mpfr_set_ld(mpfr_temp2,0.0,GMP_RNDN);
    for(i=0; i<=(N-1); i++) {
      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_tanh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) FF_gamma_r_pprime(2*i);
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_add(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);

      ld_temp = (long double) (0.5*M*FF_gamma_r(2*i));
      mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
      mpfr_sinh(mpfr_temp3,mpfr_temp3,GMP_RNDN);
      mpfr_mul(mpfr_temp3,mpfr_temp3,mpfr_temp3,GMP_RNDN);
      ld_temp = (long double) (0.5*M*FF_gamma_r_prime(2*i)*FF_gamma_r_prime(2*i));
      mpfr_set_ld(mpfr_temp4,ld_temp,GMP_RNDN);
      mpfr_div(mpfr_temp3,mpfr_temp4,mpfr_temp3,GMP_RNDN);
      mpfr_sub(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    }
    ld_temp = (long double) (0.5*M);
    mpfr_set_ld(mpfr_temp3,ld_temp,GMP_RNDN);
    mpfr_mul(mpfr_temp2,mpfr_temp2,mpfr_temp3,GMP_RNDN);
    
    mpfr_add((*result),mpfr_temp1,mpfr_temp2,GMP_RNDN);
  }

  mpfr_clear(mpfr_temp1);
  mpfr_clear(mpfr_temp2);
  mpfr_clear(mpfr_temp3);
  mpfr_clear(mpfr_temp4);
}

int main(int argc, char *argv[]) {
  long double
    internalEnergy,
    specificHeat,
    temp;

  mpfr_t
    mpfr_Z1,
    mpfr_Z2,
    mpfr_Z3,
    mpfr_Z4,
    mpfr_Z1_p,
    mpfr_Z2_p,
    mpfr_Z3_p,
    mpfr_Z4_p,
    mpfr_Z1_pp,
    mpfr_Z2_pp,
    mpfr_Z3_pp,
    mpfr_Z4_pp,
    mpfr_sum_ZxPrime_div_Zx,
    mpfr_sum_ZxPPrime_div_Zx,
    mpfr_temp1,
    mpfr_temp2;

  /***********************************/
  /** process commandline arguments **/
  /***********************************/
  J = 1.0;
  K = (double)(argc > 1 ? J*atof(argv[1]) : J*0.5*log(1.0+sqrt(2.0)));
  M = (int)(argc > 2 ? atoi(argv[2]) : 32);
  N = (int)(argc > 3 ? atoi(argv[3]) : 32);

  /*********************************/
  /** initialize mpfr_t variables **/
  /*********************************/
  mpfr_init2(mpfr_Z1,PRECISION);
  mpfr_init2(mpfr_Z2,PRECISION);
  mpfr_init2(mpfr_Z3,PRECISION);
  mpfr_init2(mpfr_Z4,PRECISION);

  mpfr_init2(mpfr_Z1_p,PRECISION);
  mpfr_init2(mpfr_Z2_p,PRECISION);
  mpfr_init2(mpfr_Z3_p,PRECISION);
  mpfr_init2(mpfr_Z4_p,PRECISION);

  mpfr_init2(mpfr_Z1_pp,PRECISION);
  mpfr_init2(mpfr_Z2_pp,PRECISION);
  mpfr_init2(mpfr_Z3_pp,PRECISION);
  mpfr_init2(mpfr_Z4_pp,PRECISION);

  mpfr_init2(mpfr_sum_ZxPrime_div_Zx,PRECISION);
  mpfr_init2(mpfr_sum_ZxPPrime_div_Zx,PRECISION);
  mpfr_init2(mpfr_temp1,PRECISION);
  mpfr_init2(mpfr_temp2,PRECISION);

  /*****************************************/
  /** compute partial partition functions **/
  /** and derivatives                     **/
  /*****************************************/
  FF_partitionFunction_Zx(&mpfr_Z1,1);
  FF_partitionFunction_Zx(&mpfr_Z2,2);
  FF_partitionFunction_Zx(&mpfr_Z3,3);
  FF_partitionFunction_Zx(&mpfr_Z4,4);

  FF_partitionFunction_ZxPrime_div_Zx(&mpfr_Z1_p,1);
  FF_partitionFunction_ZxPrime_div_Zx(&mpfr_Z2_p,2);
  FF_partitionFunction_ZxPrime_div_Zx(&mpfr_Z3_p,3);
  FF_partitionFunction_ZxPrime_div_Zx(&mpfr_Z4_p,4);

  FF_partitionFunction_ZxPPrime_div_Zx(&mpfr_Z1_pp,1);
  FF_partitionFunction_ZxPPrime_div_Zx(&mpfr_Z2_pp,2);
  FF_partitionFunction_ZxPPrime_div_Zx(&mpfr_Z3_pp,3);
  FF_partitionFunction_ZxPPrime_div_Zx(&mpfr_Z4_pp,4);

  /**************************/
  /** compute equation (3) **/
  /**************************/
  mpfr_set_ld(mpfr_sum_ZxPrime_div_Zx,0.0,GMP_RNDN);

  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z2,mpfr_Z1,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z3,mpfr_Z1,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z4,mpfr_Z1,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z1_p,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPrime_div_Zx,mpfr_sum_ZxPrime_div_Zx,mpfr_temp1,GMP_RNDN);

  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z1,mpfr_Z2,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z3,mpfr_Z2,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z4,mpfr_Z2,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z2_p,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPrime_div_Zx,mpfr_sum_ZxPrime_div_Zx,mpfr_temp1,GMP_RNDN);

  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z1,mpfr_Z3,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z2,mpfr_Z3,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z4,mpfr_Z3,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z3_p,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPrime_div_Zx,mpfr_sum_ZxPrime_div_Zx,mpfr_temp1,GMP_RNDN);
  
  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z1,mpfr_Z4,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z2,mpfr_Z4,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z3,mpfr_Z4,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z4_p,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPrime_div_Zx,mpfr_sum_ZxPrime_div_Zx,mpfr_temp1,GMP_RNDN);

  /*****************************/
  /** compute internal energy **/
  /*****************************/
  temp = (long double) mpfr_get_ld(mpfr_sum_ZxPrime_div_Zx,GMP_RNDN);
  internalEnergy  = temp*J/(M*N);
  internalEnergy += J/tanh(2.0*K);
  internalEnergy *= -1.0;


  /**************************/
  /** compute equation (4) **/
  /**************************/
  mpfr_set_ld(mpfr_sum_ZxPPrime_div_Zx,0.0,GMP_RNDN);

  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z2,mpfr_Z1,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z3,mpfr_Z1,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z4,mpfr_Z1,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z1_pp,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPPrime_div_Zx,mpfr_sum_ZxPPrime_div_Zx,mpfr_temp1,GMP_RNDN);

  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z1,mpfr_Z2,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z3,mpfr_Z2,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z4,mpfr_Z2,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z2_pp,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPPrime_div_Zx,mpfr_sum_ZxPPrime_div_Zx,mpfr_temp1,GMP_RNDN);

  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z1,mpfr_Z3,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z2,mpfr_Z3,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z4,mpfr_Z3,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z3_pp,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPPrime_div_Zx,mpfr_sum_ZxPPrime_div_Zx,mpfr_temp1,GMP_RNDN);
  
  mpfr_set_ld(mpfr_temp1,1.0,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z1,mpfr_Z4,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z2,mpfr_Z4,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp2,mpfr_Z3,mpfr_Z4,GMP_RNDN);
  mpfr_add(mpfr_temp1,mpfr_temp1,mpfr_temp2,GMP_RNDN);
  mpfr_div(mpfr_temp1,mpfr_Z4_pp,mpfr_temp1,GMP_RNDN);
  mpfr_add(mpfr_sum_ZxPPrime_div_Zx,mpfr_sum_ZxPPrime_div_Zx,mpfr_temp1,GMP_RNDN);

  /***************************/
  /** compute specific heat **/
  /***************************/
  specificHeat = (long double) mpfr_get_ld(mpfr_sum_ZxPPrime_div_Zx,GMP_RNDN);
  specificHeat -= temp*temp;
  specificHeat *= K*K/(M*N);
  specificHeat -= 2.0*K*K/(sinh(2.0*K)*sinh(2.0*K));

  /******************************/
  /** printf results to STDOUT **/
  /******************************/
  printf("%lf\t%.15Le\t%.15Le\n",K,internalEnergy,specificHeat);

  /******************************/
  /** destroy mpfr_t variables **/
  /******************************/
  mpfr_clear(mpfr_Z1);
  mpfr_clear(mpfr_Z2);
  mpfr_clear(mpfr_Z3);
  mpfr_clear(mpfr_Z4);

  mpfr_clear(mpfr_Z1_p);
  mpfr_clear(mpfr_Z2_p);
  mpfr_clear(mpfr_Z3_p);
  mpfr_clear(mpfr_Z4_p);

  mpfr_clear(mpfr_Z1_pp);
  mpfr_clear(mpfr_Z2_pp);
  mpfr_clear(mpfr_Z3_pp);
  mpfr_clear(mpfr_Z4_pp);

  mpfr_clear(mpfr_sum_ZxPrime_div_Zx);
  mpfr_clear(mpfr_sum_ZxPPrime_div_Zx);
  mpfr_clear(mpfr_temp1);
  mpfr_clear(mpfr_temp2);

  return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static long double J;
static long double K;
static int M;
static int N;

static long double FF_gamma_r(const int r)
{
  long double
    returnValue,
    c_l;

  if(r == 0) {
    returnValue = 2.0*K+log(tanh(K));
  } else {
    c_l = (cosh(2.0*K)/tanh(2.0*K))-cos(r*M_PI/N);
    returnValue = log(c_l+sqrt(c_l*c_l-1.0));
  }

  return (long double) returnValue;
}

static long double FF_gamma_r_prime(const int r)
{
  long double
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

  return (long double) returnValue;
}

static long double FF_gamma_r_pprime(const int r)
{
  long double
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

  return (long double) returnValue;
}

static long double FF_partitionFunction_Zx(const int x)
{
  int
    i;
  long double
    returnValue = 1.0;

  if(x == 1) {
    for(i=0; i<=(N-1); i++) {
      returnValue *= (2.0*cosh(0.5*M*FF_gamma_r(2*i+1)));
    }
  } else if(x == 2) {
    for(i=0; i<=(N-1); i++) {
      returnValue *= (2.0*sinh(0.5*M*FF_gamma_r(2*i+1)));
    }
  } else if(x == 3) {
    for(i=0; i<=(N-1); i++) {
      returnValue *= (2.0*cosh(0.5*M*FF_gamma_r(2*i)));
    }
  } else if(x == 4) {
    for(i=0; i<=(N-1); i++) {
      returnValue *= (2.0*sinh(0.5*M*FF_gamma_r(2*i)));
    }
  }

  return (long double) returnValue;
}

static long double FF_partitionFunction_ZxPrime_div_Zx(const int x) 
{
  int
    i;
  long double
    returnValue = 0.0;

  if(x == 1) {
    for(i=0; i<=(N-1); i++) {
      returnValue += FF_gamma_r_prime(2*i+1)*tanh(0.5*M*FF_gamma_r(2*i+1));
    }
    returnValue *= 0.5*M;
  } else if(x == 2) {
    for(i=0; i<=(N-1); i++) {
      returnValue += FF_gamma_r_prime(2*i+1)/tanh(0.5*M*FF_gamma_r(2*i+1));
    }
    returnValue *= 0.5*M;
  } else if(x == 3) {
    for(i=0; i<=(N-1); i++) {
      returnValue += FF_gamma_r_prime(2*i)*tanh(0.5*M*FF_gamma_r(2*i));
    }
    returnValue *= 0.5*M;
  } else if(x == 4) {
    for(i=0; i<=(N-1); i++) {
      returnValue += FF_gamma_r_prime(2*i)/tanh(0.5*M*FF_gamma_r(2*i));
    }
    returnValue *= 0.5*M;
  }
  
  return (long double) returnValue;
}

static long double FF_partitionFunction_ZxPPrime_div_Zx(const int x) 
{
  int
    i;
  long double
    temp1 = 0.0,
    temp2 = 0.0,
    returnValue = 0.0;

  if(x == 1) {
    for(i=0; i<=(N-1); i++) {
      temp1 += FF_gamma_r_prime(2*i+1)*tanh(0.5*M*FF_gamma_r(2*i+1));
    }
    temp1 *= 0.5*M;
    temp1 *= temp1;
    for(i=0; i<=(N-1); i++) {
      temp2 += FF_gamma_r_pprime(2*i+1)*tanh(0.5*M*FF_gamma_r(2*i+1));
      temp2 += 0.5*M*(FF_gamma_r_prime(2*i+1)*FF_gamma_r_prime(2*i+1)/(cosh(0.5*M*FF_gamma_r(2*i+1))*cosh(0.5*M*FF_gamma_r(2*i+1))));
    }
    temp2 *= 0.5*M;

    returnValue = temp1+temp2;
  } else if(x == 2) {
    for(i=0; i<=(N-1); i++) {
      temp1 += FF_gamma_r_prime(2*i+1)/tanh(0.5*M*FF_gamma_r(2*i+1));
    }
    temp1 *= 0.5*M;
    temp1 *= temp1;
    for(i=0; i<=(N-1); i++) {
      temp2 += FF_gamma_r_pprime(2*i+1)/tanh(0.5*M*FF_gamma_r(2*i+1));
      temp2 -= 0.5*M*(FF_gamma_r_prime(2*i+1)*FF_gamma_r_prime(2*i+1)/(sinh(0.5*M*FF_gamma_r(2*i+1))*sinh(0.5*M*FF_gamma_r(2*i+1))));
    }
    temp2 *= 0.5*M;

    returnValue = temp1+temp2;
  } else if(x == 3) {
    for(i=0; i<=(N-1); i++) {
      temp1 += FF_gamma_r_prime(2*i)*tanh(0.5*M*FF_gamma_r(2*i));
    }
    temp1 *= 0.5*M;
    temp1 *= temp1;
    for(i=0; i<=(N-1); i++) {
      temp2 += FF_gamma_r_pprime(2*i)*tanh(0.5*M*FF_gamma_r(2*i));
      temp2 += 0.5*M*(FF_gamma_r_prime(2*i)*FF_gamma_r_prime(2*i)/(cosh(0.5*M*FF_gamma_r(2*i))*cosh(0.5*M*FF_gamma_r(2*i))));
    }
    temp2 *= 0.5*M;

    returnValue = temp1+temp2;
  } else if(x == 4) {
    for(i=0; i<=(N-1); i++) {
      temp1 += FF_gamma_r_prime(2*i)/tanh(0.5*M*FF_gamma_r(2*i));
    }
    temp1 *= 0.5*M;
    temp1 *= temp1;
    for(i=0; i<=(N-1); i++) {
      temp2 += FF_gamma_r_pprime(2*i)/tanh(0.5*M*FF_gamma_r(2*i));
      temp2 -= 0.5*M*(FF_gamma_r_prime(2*i)*FF_gamma_r_prime(2*i)/(sinh(0.5*M*FF_gamma_r(2*i))*sinh(0.5*M*FF_gamma_r(2*i))));
    }
    temp2 *= 0.5*M;

    returnValue = temp1+temp2;
  }
  
  return (long double) returnValue;
}

int main(int argc, char *argv[]) {
  long double
    Z1,
    Z2,
    Z3,
    Z4,
    Z1Prime_div_Z1,
    Z2Prime_div_Z2,
    Z3Prime_div_Z3,
    Z4Prime_div_Z4,
    Z1PPrime_div_Z1,
    Z2PPrime_div_Z2,
    Z3PPrime_div_Z3,
    Z4PPrime_div_Z4,
    temp,
    internalEnergy,
    specificHeat;
  

  /***********************************/
  /** process commandline arguments **/
  /***********************************/
  J = 1.0;
  K = (double)(argc > 1 ? J*atof(argv[1]) : J*0.5*log(1.0+sqrt(2.0)));
  M = (int)(argc > 2 ? atoi(argv[2]) : 32);
  N = (int)(argc > 3 ? atoi(argv[3]) : 32);

  /*****************************************/
  /** compute partial partition functions **/
  /** and derivatives                     **/
  /*****************************************/
  Z1 = FF_partitionFunction_Zx(1);
  Z2 = FF_partitionFunction_Zx(2);
  Z3 = FF_partitionFunction_Zx(3);
  Z4 = FF_partitionFunction_Zx(4);
  
  Z1Prime_div_Z1 = FF_partitionFunction_ZxPrime_div_Zx(1);
  Z2Prime_div_Z2 = FF_partitionFunction_ZxPrime_div_Zx(2);
  Z3Prime_div_Z3 = FF_partitionFunction_ZxPrime_div_Zx(3);
  Z4Prime_div_Z4 = FF_partitionFunction_ZxPrime_div_Zx(4);

  Z1PPrime_div_Z1 = FF_partitionFunction_ZxPPrime_div_Zx(1);
  Z2PPrime_div_Z2 = FF_partitionFunction_ZxPPrime_div_Zx(2);
  Z3PPrime_div_Z3 = FF_partitionFunction_ZxPPrime_div_Zx(3);
  Z4PPrime_div_Z4 = FF_partitionFunction_ZxPPrime_div_Zx(4);

  /**************************/
  /** compute equation (3) **/
  /**************************/
  temp  = Z1Prime_div_Z1/(1.0+Z2/Z1+Z3/Z1+Z4/Z1);
  temp += Z2Prime_div_Z2/(Z1/Z2+1.0+Z3/Z2+Z4/Z2);
  temp += Z3Prime_div_Z3/(Z1/Z3+Z2/Z3+1.0+Z4/Z3);
  temp += Z4Prime_div_Z4/(Z1/Z4+Z2/Z4+Z3/Z4+1.0);

  /*****************************/
  /** compute internal energy **/
  /*****************************/
  internalEnergy  = temp*J/(M*N);
  internalEnergy += J/tanh(2.0*K);
  internalEnergy *= -1.0;

  /**************************/
  /** compute equation (4) **/
  /**************************/
  specificHeat  = Z1PPrime_div_Z1/(1.0+Z2/Z1+Z3/Z1+Z4/Z1);
  specificHeat += Z2PPrime_div_Z2/(Z1/Z2+1.0+Z3/Z2+Z4/Z2);
  specificHeat += Z3PPrime_div_Z3/(Z1/Z3+Z2/Z3+1.0+Z4/Z3);
  specificHeat += Z4PPrime_div_Z4/(Z1/Z4+Z2/Z4+Z3/Z4+1.0);

  /***************************/
  /** compute specific heat **/
  /***************************/
  specificHeat -= temp*temp;
  specificHeat *= K*K/(M*N);
  specificHeat -= 2.0*K*K/(sinh(2.0*K)*sinh(2.0*K));

  /******************************/
  /** printf results to STDOUT **/
  /******************************/
  printf("%Lf\t%.15Le\t%.15Le\n",K,internalEnergy,specificHeat);

  return 0;
}


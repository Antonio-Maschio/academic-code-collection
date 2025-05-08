/*
  Assignment: Make an OpenMP parallelised wave propagation
  model for computing the seismic repsonse for a wave
  propagating through a horizontally stratified medium
*/
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <thread>
#include <complex>
#include <cmath>
#include <omp.h>

// #define M_PI 3.1415

// ======================================================
// The number of frequencies sets the cost of the problem
const long NTHREADS=1;            // number of threads
const long NFREQ=4*1024*1024;         // number of frequencies per core
// const long NFREQ=64*1024;         // number of frequencies per core

const long nfreq=NFREQ*NTHREADS;  // frequencies in spectrum

// ======================================================
template <class T> 
class NUMA_Allocator {
public:
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef T value_type;

  NUMA_Allocator() { }
  NUMA_Allocator(const NUMA_Allocator& _r) { }
  ~NUMA_Allocator() { }

  // allocate raw memory including page placement
  pointer allocate(size_t numObjects, const void *localityHint=0) {
    size_t len = numObjects * sizeof(T);
    void *m = std::malloc(len);
    char *p = static_cast<char*>(m);
    if(!omp_in_parallel()) {
    #pragma omp parallel for schedule(static)
      for(size_t i=0; i<len; i+=sizeof(T)) {
        for(size_t j=0; j<sizeof(T); ++j) {
	  p[i+j]=0;
	}
      }
    }
    
    return static_cast<T*>(m);
  }

  // free raw memory
  void deallocate(pointer ptrToMemory, size_t numObjects) {
    std::free(ptrToMemory);
  }

  // construct object at given address
  void construct(pointer p, const value_type& x) {
    new(p) value_type(x);
  }

  // destroy object at given address
  void destroy(pointer p) {
    p->~value_type();
  }

private:
  void operator=(const NUMA_Allocator&) {}
};

// shorthand name for complex number type definition below
typedef std::complex<double> Complex;

// shorthand names for vector types

// Use NUMA-aware first touch allocator
typedef std::vector<Complex, NUMA_Allocator<Complex>> ComplexVector;
typedef std::vector<double, NUMA_Allocator<double>> DoubleVector;

// Use standard allocator
// typedef std::vector<Complex> ComplexVector;
// typedef std::vector<double> DoubleVector;

// Initialize Basic Constants
const double dT=0.001;     // sampling distance
const long nsamp=2*nfreq;  // samples in seismogram

// Frequency resolution (frequency sampling distance)
double dF = 1/(nsamp*dT);

// read data file with one number per line
DoubleVector read_txt_file(std::string fname) {
    DoubleVector data;  // vector of data points
    std::string line;          // string to read in each line    
    std::ifstream file(fname); // open file
    while (std::getline(file, line))     // loop over lines until end of file
        data.push_back(std::stod(line)); // convert string to double and store in array
    return data;
}

// Cooley–Tukey FFT (in-place computation)
void fft(ComplexVector& x)
{
	const long N = x.size();
	if (N <= 1) return;

	// divide
	ComplexVector even(N/2), odd(N/2);
    
    #pragma omp taskloop shared(x, even, odd) if(N > 1024)
	for (long i=0; i<N/2; i++) {
	    even[i] = x[2*i];
	    odd[i]  = x[2*i+1];
	}
    #pragma omp taskwait

	// conquer
    #pragma omp task shared(even) if (N>1024)
	fft(even);
    #pragma omp task shared(odd) if (N>1024)
	fft(odd);

    #pragma omp taskwait

	// combine
    // #pragma omp simd
    #pragma omp taskloop shared(x, even, odd) if(N > 1024)
	for (long k = 0; k < N/2; k++)
	{
		Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
		x[k    ] = even[k] + t;
		x[k+N/2] = even[k] - t;
	}
    #pragma omp taskwait

}

// inverse fft (in-place)
void ifft(ComplexVector& x)
{
    double inv_size = 1.0 / x.size();
    #pragma omp for
    for (auto& xx: x) xx = std::conj(xx); // conjugate the input
    // #pragma omp parallel shared(x)
    // {
    #pragma omp single
	fft(x);  	   // forward fft
    // }
    #pragma omp for
    for (auto& xx: x) 
        xx = std::conj(xx)  // conjugate the output
            * inv_size;     // scale the numbers
}

// Main routine: propgate wave through layers and compute seismogram
DoubleVector propagator(DoubleVector wave,
                        DoubleVector density,
                        DoubleVector velocity) {
    const long nlayers = density.size();
    DoubleVector imp(nlayers);      // impedance
    DoubleVector ref(nlayers-1);    // reflection coefficient
    ComplexVector half_filter(nfreq/2+1,1); // half filter
    ComplexVector filter(nfreq+1);  // full filter
    DoubleVector half_wave(nfreq+1,0); // half wave
    ComplexVector wave_spectral(nsamp); // FFT(wave)
    ComplexVector U(nfreq+1,0);     // Upgoing waves
    ComplexVector Upad(nsamp,0);    // FFT(seismogram)
    DoubleVector seismogram(nsamp); // final seismogram

    long n_wave = wave.size();             // size of wave array
    long lc = std::lround(std::floor(nfreq*0.01)); // low-cut indices
    double mean_wave = 0.;                 // wave zero point

    std::chrono::time_point<std::chrono::system_clock> tstart1,tstart2,tend1,tend2;

    auto tstart = std::chrono::high_resolution_clock::now(); // start time (nano-seconds)
    

        // Compute seismic impedance

    for (long i=0; i < nlayers; i++)
        imp[i] = density[i] * velocity[i];
    
    // Reflection coefficients at the base of the layers :
    for (long i=0; i < nlayers-1; i++)
        ref[i] = (imp[i+1] - imp[i])/(imp[i+1] + imp[i]);

    // Spectral window (both low- and high cut)
    for (long i=0; i < lc+1; i++)
        half_filter[i]= (sin(M_PI*(2*i-lc)/(2*lc)))/2+0.5;

    for (long i=0; i < nfreq/2+1; i++)
        filter[i] = half_filter[i];

    filter[nfreq/2+1] = 1;

    for (long i=nfreq/2+2; i < nfreq+1; i++)
        filter[i] = half_filter[nfreq+1-i];

    for (long i=0; i < n_wave/2; i++)
        half_wave[i] = wave[n_wave/2-1+i];

    for (long i=0; i < 2*nfreq; i++) {
        if (i < nfreq) {
            wave_spectral[i] = half_wave[i];
        } else {
            wave_spectral[i] = half_wave[2*nfreq-i];
        }
        mean_wave += std::real(wave_spectral[i]);
    }

    mean_wave = mean_wave / nsamp;

    for (long i=0.; i < 2*nfreq; i++)
        wave_spectral[i] -= mean_wave;

    // Fourier transform waveform to frequency domain
    tstart1 = std::chrono::high_resolution_clock::now(); // start time (nano-seconds)
    #pragma omp parallel shared(wave_spectral) // firstprivate(filter) // shared U
    {
    #pragma omp single // nowait
    fft(wave_spectral);
    // }
    tend1 = std::chrono::high_resolution_clock::now(); // end time (nano-seconds)

    // spectrum U of upgoing waves just below the surface.
    // See eq. (43) and (44) in Ganley (1981).
// #
    #pragma omp for
    for (long i=0; i < nfreq+1; i++) {
        Complex omega{0, 2*M_PI*i*dF};
        Complex exp_omega = exp( - dT * omega);
        Complex Y = 0;
        for (long n=nlayers-2; n > -1; n--)
            Y = exp_omega * (ref[n] + Y) / (1.0 + ref[n]*Y);
        U[i] = Y;
    }

    // Compute seismogram
    #pragma omp for
    for (long i=0; i < nfreq+1; i++) {
        U[i] *= filter[i];
        Upad[i] = U[i];
    }
    #pragma omp for
    for (long i=nfreq+1; i < nsamp; i++)
        Upad[i] = std::conj(Upad[nsamp - i]);
    #pragma omp for
    for (long i=0; i < nsamp; i++)
        Upad[i] *= wave_spectral[i];
    
    // Fourier transform back again
    tstart2 = std::chrono::high_resolution_clock::now(); // start time (nano-seconds)
    ifft(Upad);
    tend2 = std::chrono::high_resolution_clock::now(); // end time (nano-seconds)
    #pragma omp for
    for (long i=0; i < nsamp; i++)
        seismogram[i] = std::real(Upad[i]);
    
    #pragma omp single
    {
    auto tend = std::chrono::high_resolution_clock::now(); // end time (nano-seconds)

    std::cout <<  "Wave zero-point        : "  << std::setw(9) << std::setprecision(5) 
              << mean_wave<< "\n";    
    std::cout <<  "Seismogram first coeff : "  << std::setw(9) << std::setprecision(5) 
              << seismogram[0] << ", " << seismogram[1] << ", " << seismogram[2] << ", " << seismogram[3] <<"\n";    
    std::cout <<  "Elapsed time for FFTs  :" << std::setw(9) << std::setprecision(4)
              << (tend1 - tstart1 + tend2 - tstart2).count()*1e-9 << "\n";
    std::cout <<  "Elapsed time without FFTs:" << std::setw(9) << std::setprecision(4)
              << (tend - tstart - (tend1 - tstart1 + tend2 - tstart2)).count()*1e-9 << "\n";
    std::cout <<  "Elapsed time:" << std::setw(9) << std::setprecision(4)
              << (tend - tstart).count()*1e-9 << "\n";
    }
    }//parallelregion

    
    return seismogram;
}

//======================================================================================================
//======================== Main function ===============================================================
//======================================================================================================
int main(int argc, char* argv[]){    
    // Load the wave profile and the density and velocity structure of the rock from text files
    DoubleVector wave = read_txt_file("wave_data.txt");         // input impulse wave in medium
    DoubleVector density = read_txt_file("density_data.txt");   // density as a function of depth
    DoubleVector velocity = read_txt_file("velocity_data.txt"); // seismic wave velocity as a function of depth
//   omp_set_num_threads(8);

    // Propagate wave
    auto seismogram = propagator(wave,density,velocity);
    
    // write output and make checksum
    double checksum=0;
    std::ofstream file("seismogram.txt"); // open file
    for (long i=0; i < nsamp; i++) {
        file << seismogram[i] << '\n';
        checksum += std::abs(seismogram[i]);
    }
    std::cout <<  "Checksum    :" << std::setw(20) << std::setprecision(15)
              << checksum << "\n";
}

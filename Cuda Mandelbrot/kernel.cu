#include <SFML/Graphics.hpp>
#include <ccomplex>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Color
{
	char r, g, b, a;
};


__global__ void render (Color * canvas, double offsetx, double offsety, double zoom);
__device__ int mandelbrot (double startReal, double startImag, int maximum);
__device__ Color spectrum (double n);

int const threads = 64; 

int main ()
{
	sf::RenderWindow window (sf::VideoMode (1920, 1080), "Mandelbrot", sf::Style::Fullscreen);

	Color * video_canvas;
	cudaMalloc (&video_canvas, 1920 * 1080 * 4);

	sf::Color * canvas = new sf::Color[1920 * 1080];

	sf::Texture texture;
	texture.create (1920, 1080);

	double offset_x = 0;
	double offset_y = 0;
	double zoom = 1;

	while (window.isOpen ())
	{
		sf::Event event;
		while (window.pollEvent (event))
		{
			switch (event.type)
			{
			default:
				break;

			case sf::Event::Closed:
				window.close ();
			}
		}
		
		if (sf::Keyboard::isKeyPressed (sf::Keyboard::Dash)) zoom /= 0.94;
		if (sf::Keyboard::isKeyPressed (sf::Keyboard::Equal)) zoom *= 0.94;
		if (sf::Keyboard::isKeyPressed (sf::Keyboard::A)) offset_x -= 0.01 * zoom;
		if (sf::Keyboard::isKeyPressed (sf::Keyboard::D)) offset_x += 0.01 * zoom;
		if (sf::Keyboard::isKeyPressed (sf::Keyboard::S)) offset_y += 0.01 * zoom;
		if (sf::Keyboard::isKeyPressed (sf::Keyboard::W)) offset_y -= 0.01 * zoom;


		if (sf::Keyboard::isKeyPressed (sf::Keyboard::Escape)) window.close();

		render <<< 1920 * 1080 / threads, threads >>> (video_canvas, offset_x, offset_y, zoom);

		cudaMemcpy (canvas, video_canvas, 1920 * 1080 * 4, cudaMemcpyDeviceToHost);
		
		texture.update (reinterpret_cast<sf::Uint8 *>(canvas), 1920, 1080, 0, 0);

		window.draw (sf::Sprite (texture));
		window.display ();
	}

	delete[] canvas;
	cudaFree (video_canvas);
}

__global__ void render (Color * canvas, double offsetx, double offsety, double zoom)
{
	int const i = blockIdx.x * blockDim.x + threadIdx.x;
	int2 const p = { i % 1920, i / 1920 };

	int limit = 128;
	int limit_sqrt = sqrtf (limit);

	double f_x = (static_cast <double> (p.x) / 1080 * 2 - 1) * zoom + offsetx;
	double f_y = (static_cast <double> (p.y) / 1090 * 2 - 1) * zoom + offsety;

	int iters = mandelbrot (f_x, f_y, limit);

	Color color = spectrum (iters);

	canvas[1920 * p.y + p.x] = color;
}

__device__ int mandelbrot (double startReal, double startImag, int maximum) {
	int counter = 0;
	double zReal = startReal;
	double zImag = startImag;
	double nextRe;

	while (zReal * zReal + zImag * zImag <= 4.0 && counter <= maximum) {
		nextRe = zReal * zReal - zImag * zImag + startReal;
		zImag = 2.0 * zReal * zImag + startImag;
		zReal = nextRe;
		if (zReal == startReal && zImag == startImag) { // a repetition indicates that the point is in the Mandelbrot set
			return -1; // points in the Mandelbrot set are represented by a return value of -1
		}
		counter += 1;
	}

	if (counter >= maximum) {
		return -1; // -1 is used here to indicate that the point lies within the Mandelbrot set
	}
	else {
		return counter; // returning the number of iterations allows for colouring
	}
}

__device__ Color spectrum (double iterations)
{
	int r, g, b;

	if (iterations == -1) {
		r = 0;
		g = 0;
		b = 0;
	}
	else if (iterations == 0) {
		r = 255;
		g = 0;
		b = 0;
	}
	else {
		// colour gradient:      Red -> Blue -> Green -> Red -> Black
		// corresponding values:  0  ->  16  ->  32   -> 64  ->  127 (or -1)
		if (iterations < 16) {
			r = 16 * (16 - iterations);
			g = 0;
			b = 16 * iterations - 1;
		}
		else if (iterations < 32) {
			r = 0;
			g = 16 * (iterations - 16);
			b = 16 * (32 - iterations) - 1;
		}
		else if (iterations < 64) {
			r = 8 * (iterations - 32);
			g = 8 * (64 - iterations) - 1;
			b = 0;
		}
		else { // range is 64 - 127
			r = 255 - (iterations - 64) * 4;
			g = 0;
			b = 0;
		}
	}

	return { r, g, b, 255 };
}

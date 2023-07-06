/*
compile with
`g++ mandelbrot_sse_avx.cpp -Wall -Werror -Wpedantic -O3 -mavx2 -fopenmp -lsfml-graphics -lsfml-system -lsfml-window`
*/

#if defined(_WIN32) || !defined(_WIN64) ||                             \
    (defined(__CYGWIN__) && !defined(_WIN32)) || defined(__linux__) || \
    defined(linux) || defined(__linux)
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#else
#error windows or linux required
#endif

#include <immintrin.h>

namespace xcr {
/*
 * Constants for the (0, 0) point and window parameters
 */
const float O_x = -1.325f, O_y = 0;

const size_t WINDOW_WIDTH = 800;
const size_t WINDOW_HEIGHT = 600;

typedef struct {
    const float dx = 1 / 800.f;
    const float dy = 1 / 800.f;
    const float max_points = 256;

    const __m256 maxR_vector = _mm256_set1_ps(100.f);
    const __m256 _255 = _mm256_set1_ps(255.f);
    const __m256 _76543210 =
        _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    const __m256 maxP_vector = _mm256_set1_ps(max_points);

    // current point coordinates in set
    float cur_x = 0.f;
    float cur_y = 0.f;
    // current scale
    float scale = 1.f;
} SSEdata;
}  // namespace xcr

void ProcessSetState(xcr::SSEdata& data,
                     sf::Image& image,
                     int iy,
                     bool& do_process_set_state) {
    /**
     * ! Formula for new y0 coord:
     * ! We are decreasing iy by WINDOW_HEIGHT / 2 to find, range of
     * point's coordinates on the screen ! 0_y + cur_y are added to find
     * the actual location based on the previous coordinates
     *
     * ? x0 is calculated inside of the cycle because OpenMP does not
     * allow non-constant variable usage in cycle ?
     */

    float y0 =
        ((static_cast<float>(iy) - static_cast<float>(xcr::WINDOW_HEIGHT) / 2) *
         data.dy) *
            data.scale +
        xcr::O_y + data.cur_y;

#pragma omp parallel for
    for (size_t ix = 0; ix < xcr::WINDOW_WIDTH;
         ix += 8 /*, x0 += dx * 8 * scale*/) {
        float x0 = ((static_cast<float>(ix) -
                     static_cast<float>(xcr::WINDOW_WIDTH) / 2) *
                    data.dx) *
                       data.scale +
                   xcr::O_x + data.cur_x;

        /**
         * * Formula: *
         * ! X0_vector = x0 + dx * 8 * scale
         * ? Y0_vector = y0 (doesn't change right away) ?
         */
        __m256 X0_vector =
            _mm256_add_ps(_mm256_set1_ps(x0),
                          _mm256_mul_ps(data._76543210,
                                        _mm256_set1_ps(data.dx * data.scale)));
        __m256 Y0_vector = _mm256_set1_ps(y0);

        __m256 X_vector = X0_vector, Y_vector = Y0_vector;

        __m256i iter_vector = _mm256_setzero_si256();  // Setting all elements
                                                       // of iter_vector to zero

        bool do_process_points = true;
        for (int point = 0; point < data.max_points; ++point) {
            if (do_process_points) {
                __m256 x2 = _mm256_mul_ps(X_vector, X_vector),
                       y2 = _mm256_mul_ps(Y_vector, Y_vector);

                __m256 r2 = _mm256_add_ps(x2, y2);
                /**
                 * ! _CMP_LE_OQ or _CMP_LE_OS is a defined operator <=
                 * in AVX ! ? _LE_ means less-equal ?
                 */
                __m256 cmp = _mm256_cmp_ps(r2, data.maxR_vector, _CMP_LE_OQ);

                /**
                 * ! mask will contain '1' byte, if cmp (vector)
                 * contains not-null object, else 0 ! ? if !mask <=>
                 * mask == 0, then all elements in vector cmp are null ?
                 */
                int mask = _mm256_movemask_ps(cmp);
                if (mask == 0) {
                    do_process_points = 0;
                }

                iter_vector =
                    _mm256_sub_epi32(iter_vector, _mm256_castps_si256(cmp));

                __m256 xy = _mm256_mul_ps(X_vector, Y_vector);

                X_vector = _mm256_add_ps(_mm256_sub_ps(x2, y2), X0_vector);
                Y_vector = _mm256_add_ps(_mm256_add_ps(xy, xy), Y0_vector);
            }
        }

        /**
         * * Filter mask with the special formula *
         * ? Filtered mask allows us to make better coloring algorithm ?
         */
        __m256 I = _mm256_mul_ps(
            _mm256_sqrt_ps(_mm256_sqrt_ps(_mm256_div_ps(
                _mm256_cvtepi32_ps(iter_vector), data.maxP_vector))),
            data._255);

        for (int point_index = 0; point_index < 8; point_index++) {
            int* pn = reinterpret_cast<int*>(&iter_vector);
            float* pI = reinterpret_cast<float*>(&I);

            auto c = static_cast<uint8_t>(pI[point_index]);
            image.setPixel(iy, ix + point_index,
                           pn[point_index] < data.max_points
                               ? sf::Color({sf::Uint8(255 - 2 * c),
                                            sf::Uint8(c % 2 * 64), c})
                               : sf::Color());
        }
    }
}

void FillSet(sf::RenderWindow& window,
             sf::Image& image,
             sf::Texture& texture,
             sf::Sprite& sprite) {
    xcr::SSEdata data;
    bool do_process_set_state = true;

    while (window.isOpen()) {
        sf::Event current_event;
        while (window.pollEvent(current_event)) {
            if (current_event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                do_process_set_state = false;
                window.close();
            }

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
                data.cur_x +=
                    data.dx * (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)
                                   ? 50.f
                                   : 5.f);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
                data.cur_x -=
                    data.dx * (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)
                                   ? 50.f
                                   : 5.f);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
                data.cur_y -=
                    data.dy * (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)
                                   ? 50.f
                                   : 5.f);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
                data.cur_y +=
                    data.dy * (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)
                                   ? 50.f
                                   : 5.f);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
                data.scale +=
                    data.dx * (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)
                                   ? 50.f
                                   : 5.f);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
                data.scale -=
                    data.dx * (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift)
                                   ? 50.f
                                   : 5.f);
        }

#pragma omp parallel for
        for (size_t iy = 0; iy < xcr::WINDOW_HEIGHT * do_process_set_state;
             ++iy) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                do_process_set_state = false;
            }

            ProcessSetState(data, image, iy, do_process_set_state);
        }

        texture.loadFromImage(image);
        sprite.setTexture(texture);

        window.draw(sprite);

        window.display();
    }
}

int main() {
    sf::RenderWindow main_window(
        sf::VideoMode(xcr::WINDOW_WIDTH, xcr::WINDOW_HEIGHT),
        "Mandelbrot set + SSE");
    sf::Image set_image;
    set_image.create(main_window.getSize().x, main_window.getSize().y,
                     sf::Color::Black);

    sf::Texture set_texture;
    sf::Sprite image_sprite;

    FillSet(main_window, set_image, set_texture, image_sprite);
    return 0;
}
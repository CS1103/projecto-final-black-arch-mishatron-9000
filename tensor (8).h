//
// Created by Usuario on 10/06/2025.
//



#ifndef EPIC1_OFICIAL_TENSOR_H
#define EPIC1_OFICIAL_TENSOR_H

#include <array>
#include <vector>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>
#include <type_traits>
#include <string>

namespace utec::algebra {

    template <typename T, std::size_t Rank>
    class Tensor {
    public:
        using Shape = std::array<std::size_t, Rank>;

        // 1) Constructor por defecto
        Tensor() noexcept
                : shape_{}, strides_{}, data_{} {}

        // 2) Construye a partir de Shape (std::array)
        explicit Tensor(const Shape& shape)
                : shape_(shape),
                  strides_(make_strides(shape_)) {
            data_.resize(num_elems(shape_));
        }

        // 3) Construye a partir de N dimensiones (N == Rank)
        template <typename... Dims>
        explicit Tensor(Dims... dims) {
            if constexpr (sizeof...(Dims) == Rank) {
                shape_   = { static_cast<std::size_t>(dims)... };
                strides_ = make_strides(shape_);
                data_.resize(num_elems(shape_));
            } else {
                throw std::invalid_argument(
                        "Number of dimensions do not match with " + std::to_string(Rank));
            }
        }

        // Acceso variádico
        template <typename... Idxs, typename = std::enable_if_t<sizeof...(Idxs) == Rank>>
        T& operator()(Idxs... idxs) {
            Shape idx{ static_cast<std::size_t>(idxs)... };
            return data_[linear_offset(idx)];
        }
        template <typename... Idxs, typename = std::enable_if_t<sizeof...(Idxs) == Rank>>
        const T& operator()(Idxs... idxs) const {
            Shape idx{ static_cast<std::size_t>(idxs)... };
            return data_[linear_offset(idx)];
        }

        // Iteradores
        auto begin() noexcept        { return data_.begin(); }
        auto end() noexcept          { return data_.end();   }
        auto cbegin() const noexcept { return data_.cbegin(); }
        auto cend()   const noexcept { return data_.cend();   }

        // Nuevo: tamaño total de elementos
        std::size_t size() const noexcept { return data_.size(); }

        // Forma actual
        const Shape& shape() const noexcept { return shape_; }

        // Redimensionar
        void reshape(const Shape& new_shape) {
            if (new_shape.size() != Rank)
                throw std::invalid_argument(
                        "Number of dimensions do not match with " + std::to_string(Rank));
            shape_   = new_shape;
            strides_ = make_strides(shape_);
            data_.resize(num_elems(shape_));
        }
        template <typename... Dims>
        void reshape(Dims... dims) {
            if constexpr (sizeof...(Dims) == Rank) {
                reshape(Shape{ static_cast<std::size_t>(dims)... });
            } else {
                throw std::invalid_argument(
                        "Number of dimensions do not match with " + std::to_string(Rank));
            }
        }

        // Rellenar con un valor
        void fill(const T& value) noexcept {
            std::fill(data_.begin(), data_.end(), value);
        }

        // Asignar desde initializer_list
        Tensor& operator=(std::initializer_list<T> list) {
            if (list.size() != data_.size())
                throw std::invalid_argument("Data size does not match tensor size");
            std::copy(list.begin(), list.end(), data_.begin());
            return *this;
        }

        // Operación elemento a elemento con broadcasting
        Tensor elementwise_op(const Tensor& other, auto op) const {
            Shape rshape;
            for (std::size_t i = 0; i < Rank; ++i) {
                auto s1 = shape_[i], s2 = other.shape_[i];
                if      (s1 == s2) rshape[i] = s1;
                else if (s1 == 1)  rshape[i] = s2;
                else if (s2 == 1)  rshape[i] = s1;
                else throw std::invalid_argument(
                            "Shapes do not match and are not compatible for broadcasting");
            }
            Tensor result(rshape);
            std::array<std::size_t, Rank> idx;
            for (std::size_t lin = 0, tot = result.data_.size(); lin < tot; ++lin) {
                std::size_t rem = lin;
                for (std::size_t i = 0; i < Rank; ++i) {
                    idx[i] = rem / result.strides_[i];
                    rem    %= result.strides_[i];
                }
                std::size_t off1 = 0, off2 = 0;
                for (std::size_t i = 0; i < Rank; ++i) {
                    auto id = (shape_[i] == 1 ? 0 : idx[i]);
                    off1 += id * strides_[i];
                    auto jd = (other.shape_[i] == 1 ? 0 : idx[i]);
                    off2 += jd * other.strides_[i];
                }
                result.data_[lin] = op(data_[off1], other.data_[off2]);
            }
            return result;
        }

        Tensor operator+(const Tensor& o) const { return elementwise_op(o, std::plus<>()); }
        Tensor operator-(const Tensor& o) const { return elementwise_op(o, std::minus<>()); }
        Tensor operator*(const Tensor& o) const { return elementwise_op(o, std::multiplies<>()); }

        // Operaciones escalares
        Tensor operator*(const T& s) const {
            Tensor r(shape_);
            for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * s;
            return r;
        }
        Tensor operator/(const T& s) const {
            Tensor r(shape_);
            for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] / s;
            return r;
        }
        Tensor operator+(const T& s) const noexcept {
            Tensor r(shape_);
            for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + s;
            return r;
        }
        Tensor operator-(const T& s) const noexcept {
            Tensor r(shape_);
            for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - s;
            return r;
        }

        // Transpuesta 2D (swap de últimos dos ejes)
        Tensor transpose_2d() const {
            if constexpr (Rank < 2)
                throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
            Shape nshape = shape_;
            std::swap(nshape[Rank-2], nshape[Rank-1]);
            Tensor r(nshape);
            std::array<std::size_t, Rank> idx;
            for (std::size_t lin = 0; lin < data_.size(); ++lin) {
                std::size_t rem = lin;
                for (std::size_t i = 0; i < Rank; ++i) {
                    idx[i] = rem / strides_[i];
                    rem    %= strides_[i];
                }
                std::swap(idx[Rank-2], idx[Rank-1]);
                r.data_[r.linear_offset(idx)] = data_[lin];
            }
            return r;
        }

        // Impresión anidada
        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            print_recursive(os, t, 0, 0);
            return os;
        }

    private:
        Shape          shape_{}, strides_{};
        std::vector<T> data_;

        static constexpr std::size_t num_elems(const Shape& s) {
            std::size_t n = 1;
            for (auto v : s) n *= v;
            return n;
        }
        static constexpr Shape make_strides(const Shape& s) {
            Shape st{};
            std::size_t acc = 1;
            for (std::size_t i = Rank; i-- > 0;) {
                st[i] = acc;
                acc  *= s[i];
            }
            return st;
        }
        std::size_t linear_offset(const Shape& idx) const {
            std::size_t off = 0;
            for (std::size_t i = 0; i < Rank; ++i)
                off += idx[i] * strides_[i];
            return off;
        }
        static void print_recursive(std::ostream& os, const Tensor& t,
                                    std::size_t dim, std::size_t offset) {
            if (dim < Rank - 1) {
                os << "{\n";
                for (std::size_t i = 0; i < t.shape_[dim]; ++i)
                    print_recursive(os, t, dim + 1, offset + i * t.strides_[dim]);
                if (dim > 0) os << "}\n"; else os << "}";
            } else {
                for (std::size_t j = 0; j < t.shape_[dim]; ++j) {
                    os << t.data_[offset + j * t.strides_[dim]];
                    if (j + 1 < t.shape_[dim]) os << ' ';
                }
                os << '\n';
            }
        }
    };

    // Funciones auxiliares
    template <typename T, std::size_t R>
    Tensor<T,R> transpose_2d(const Tensor<T,R>& t) { return t.transpose_2d(); }

    template <typename T>
    Tensor<T,2> matrix_product(const Tensor<T,2>& a,
                               const Tensor<T,2>& b) {
        auto ash = a.shape(), bsh = b.shape();
        size_t M = ash[0], K = ash[1], K2 = bsh[0], N = bsh[1];
        if (K != K2)
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        Tensor<T,2> r(M, N);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
                T sum{};
                for (size_t k = 0; k < K; ++k)
                    sum += a(i,k) * b(k,j);
                r(i,j) = sum;
            }
        return r;
    }

    template <typename T>
    Tensor<T,3> matrix_product(const Tensor<T,3>& a,
                               const Tensor<T,3>& b) {
        auto ash = a.shape(), bsh = b.shape();
        size_t B = ash[0], M = ash[1], K = ash[2];
        size_t B2 = bsh[0], K2 = bsh[1], N = bsh[2];
        if (K != K2)
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        if (B != B2)
            throw std::invalid_argument("Batch dimensions do not match for multiplication");
        Tensor<T,3> r(B, M, N);
        for (size_t batch = 0; batch < B; ++batch)
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j) {
                    T sum{};
                    for (size_t k = 0; k < K; ++k)
                        sum += a(batch,i,k) * b(batch,k,j);
                    r(batch,i,j) = sum;
                }
        return r;
    }

    // CTAD
    template<typename... Dims>
    Tensor(Dims...)-> Tensor<std::common_type_t<Dims...>, sizeof...(Dims)>;
    template<typename U, std::size_t R>
    Tensor(const std::array<U,R>&)-> Tensor<U,R>;

}



#endif //EPIC1_OFICIAL_TENSOR_H

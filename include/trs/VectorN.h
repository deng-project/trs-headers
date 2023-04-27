#ifndef VECTORN_H
#define VECTORN_H

#include <type_traits>
#include <vector>
#include <cmath>

namespace TRS {

	template<typename T>
	class MatrixN;

	template<typename T>
	class VectorN : public std::vector<T> {
		public:
			VectorN() : vector<T>() {}
			VectorN(const VectorN&) = default;
			VectorN(VectorN&&) noexcept = default;

			void operator=(const VectorN& v) {
				resize(v.size());
				for (size_t i = 0; i < size(); i++)
					(*this)[i] = v[i];
			}

			explicit VectorN(size_t _count, const T& _value = T()) :
				vector<T>(_count, _value) {}

			VectorN(std::initializer_list<T> _init) :
				vector<T>(_init) {}

			
			
			bool operator==(const VectorN<T>& v2) const {
				bool isEqual = size() == v2.size();

				if (isEqual) {
					for (size_t i = 0; i < size(); i++) {
						isEqual = (*this)[i] == v2[i];

						if (!isEqual) break;
					}
				}

				return isEqual;
			}

			bool operator!=(const VectorN<T>& v2) const {
				return !(*this == v2);
			}

			// addition
			VectorN<T> operator+(const VectorN<T>& v2) const {
				VectorN<T> result(std::min(size(), v2.size()));
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < std::min(size(), v2.size()); i++) {
						result[i] = (*this)[i] + v2[i];
					}
				}
				return result;
			}

			// subtraction
			VectorN<T> operator-(const VectorN<T>& v2) const {
				VectorN<T> result(std::min(size(), v2.size()));
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < std::min(size(), v2.size()); i++) {
						result[i] = (*this)[i] - v2[i];
					}
				}

				return result;
			}

			// multiplication with a scalar
			VectorN<T> operator*(const T v) const {
				VectorN<T> result(*this);

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++)
						result[i] *= v;
				}

				return result;
			}

			// division with a scalar
			VectorN<T> operator/(const T v) const {
				VectorN<T> result(*this);
				
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++)
						result[i] /= v;
				}
			
				return result;
			}

			// unary - operation
			VectorN<T> operator-() const {
				VectorN<T> result = *this;

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++)
						result[i] = -result[i];
				}

				return result;
			}

			T Length() const {
				T sig = T();
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++)
						sig += (*this)[i] * (*this)[i];

					return std::sqrtf(sig);
				}

				return sig;
			}

			void Normalise() {
				if constexpr (std::is_arithmetic<T>::value) {
					T len = Length();

					if (len > 0) {
						for (size_t i = 0; i < size(); i++)
							(*this)[i] /= len;
					}
				}
			}

			// dot product
			T operator*(const VectorN<T>& v2) const {
				T dotProduct = T();
				if constexpr (std::is_arithmetic<T>::value) {
					if (size() == v2.size()) {
						for (size_t i = 0; i < size(); i++) {
							dotProduct += (*this)[i] * v2[i];
						}
					}
				}

				return dotProduct;
			}

			// multiplication with matrix
			VectorN<T> operator*(const MatrixN<T>& mat) const;
			
			// v * v.Transpose()
			MatrixN<T> ExpandToMatrix();
	};
}

#endif
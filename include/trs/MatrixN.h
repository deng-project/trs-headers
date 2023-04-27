#ifndef MATRIXN_H
#define MATRIXN_H

#include <type_traits>
#include <vector>
#include <trs/VectorN.h>

namespace TRS {
	template<typename T>
	class MatrixN : public std::vector<std::vector<T>> {
		public:
			MatrixN() : std::vector<std::vector<T>>() {}
			MatrixN(const MatrixN& m) {
				resize(m.size());
				for (auto it = begin(); it != end(); it++)
					it->resize(m.size());

				for (size_t i = 0; i < size(); i++)
					for (size_t j = 0; j < size(); j++)
						(*this)[i][j] = m[i][j];
			};
			MatrixN(MatrixN&&) noexcept = default;
			
			void operator=(const MatrixN& m) {
				resize(m.size());

				for (size_t i = 0; i < size(); i++) {
					(*this)[i].resize(m.size());
					for (size_t j = 0; j < (*this)[i].size(); j++)
						(*this)[i][j] = m[i][j];
				}
			}

			explicit MatrixN(size_t _count, const T& _value = T()) :
				std::vector<std::vector<T>>(_count) 
			{
				for (size_t i = 0; i < size(); i++)
					(*this)[i].resize(size(), _value);
			}

			MatrixN(std::initializer_list<std::vector<T>> _init) :
				std::vector<std::vector<T>>(_init) {}

			static MatrixN<T> MakeIdentity(size_t n) {
				MatrixN<T> matrix(n);
				
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < n; i++)
						matrix[i][i] = static_cast<T>(1);
				}

				return matrix;
			}

			void operator+=(const MatrixN<T>& m) {
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++)
						for (size_t j = 0; j < (*this)[i].size(); j++)
							(*this)[i][j] += m[i][j];
				}
			}

			void operator-=(const MatrixN<T>& m) {
				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++)
						for (size_t j = 0; j < (*this)[i].size(); j++)
							(*this)[i][j] -= m[i][j];
				}
			}

			// multiplication with a scalar
			MatrixN<T> operator*(const T v) const {
				MatrixN<T> result(*this);

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < result.size(); i++)
						for (size_t j = 0; j < result[i].size(); j++)
							result[i][j] *= v;
				}

				return result;
			}

			// division with a scalar
			MatrixN<T> operator/(const T v) const {
				MatrixN<T> result(*this);

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < result.size(); i++)
						for (size_t j = 0; j < result[i].size(); j++)
							result[i][j] /= v;
				}

				return result;
			}

			// unary sign change
			MatrixN<T> operator-() const {
				MatrixN<T> result(*this);

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < result.size(); i++)
						for (size_t j = 0; j < result[i].size(); j++)
							result[i][j] = -result[i][j];
				}

				return result;
			}


			// matrix addition
			MatrixN<T> operator+(const MatrixN<T>& m2) const {
				MatrixN<T> result(*this);

				if constexpr (std::is_arithmetic<T>::value) {
					if (size() == m2.size()) {
						for (size_t i = 0; i < result.size(); i++)
							for (size_t j = 0; j < result[i].size(); j++)
								result[i][j] += m2[i][j];
					}
				}

				return result;
			}

			// matrix subtraction
			MatrixN<T> operator-(const MatrixN<T>& m2) const {
				MatrixN<T> result(*this);

				if constexpr (std::is_arithmetic<T>::value) {
					if (size() == m2.size()) {
						for (size_t i = 0; i < result.size(); i++)
							for (size_t j = 0; j < result[i].size(); j++)
								result[i][j] -= m2[i][j];
					}
				}

				return result;
			}


			// matrix multiplication
			MatrixN<T> operator*(const MatrixN<T>& m2) const {
				MatrixN<T> result(size());

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < result.size(); i++) {
						for (size_t j = 0; j < result[i].size(); j++) {
							result[i][j] = 0;
							for (size_t k = 0; k < size(); k++) {
								result[i][j] += (*this)[i][k] * m2[k][j];
							}
						}
					}
				}

				return result;
			}

			// matrix multiplication with a column vector
			VectorN<T> operator*(const VectorN<T>& v1) const {
				VectorN<T> result(size());

				if constexpr (std::is_arithmetic<T>::value) {
					for (size_t i = 0; i < size(); i++) {
						for (size_t j = 0; j < size(); j++)
							result[i] += (*this)[i][j] * v1[j];
					}
				}

				return result;
			}

			T Determinant() const {
				T det = T();

				if constexpr (std::is_arithmetic<T>::value) {
					MatrixN<T> upper(size()), lower(size());

					for (size_t i = 0; i < size(); i++) {
						// upper triangular
						for (size_t j = i; j < size(); j++) {
							T sum = T();
							for (size_t k = 0; k < i; k++)
								sum += lower[i][k] * upper[k][j];

							upper[i][j] = (*this)[i][j] - sum;
						}

						// lower triangular
						for (size_t j = i; j < size(); j++) {
							if (i == j)
								lower[i][i] = static_cast<T>(1);
							else {
								T sum = T();
								for (size_t k = 0; k < i; k++)
									sum += (lower[j][k] * upper[k][i]);

								if (!upper[i][i])
									return static_cast<T>(0);
								else
									lower[j][i] = ((*this)[j][i] - sum) / upper[i][i];
							}
						}
					}

					det = static_cast<T>(1);
					for (size_t i = 0; i < upper.size(); i++)
						det *= upper[i][i];
				}

				return det;
			}

		MatrixN<T> Inverse() const {
			MatrixN<T> inv(size());

			if constexpr (std::is_arithmetic<T>::value) {
				MatrixN<T> upper(size()), lower(size());

				// find LU matrices
				for (size_t i = 0; i < size(); i++) {
					// upper triangular
					for (size_t j = 0; j < size(); j++) {
						T sum = T();
						for (size_t k = 0; k < i; k++)
							sum += lower[i][k] * upper[k][j];

						upper[i][j] = (*this)[i][j] - sum;
					}

					// lower triangular
					for (size_t j = 0; j < size(); j++) {
						if (i == j)
							lower[i][j] = static_cast<T>(1);
						else {
							T sum = T();
							for (size_t k = 0; k < i; k++)
								sum += (lower[j][k] * upper[k][i]);

							lower[j][i] = ((*this)[j][i] - sum) / upper[i][i];
						}
					}
				}

				// calculate inverse of these matrices
				MatrixN<T> inverseUpper(size()), inverseLower(size());
					
				// lower triangular
				for (size_t i = 0; i < size(); i++) {
					for (size_t j = 0; j <= i; j++) {
						T sum = static_cast<T>(0);
						for (size_t k = 0; k < i; k++)
							sum += lower[i][k] * inverseLower[k][j];

						float b = 0.0f;
						if (i == j)
							b = 1.0f;

						inverseLower[i][j] = (static_cast<T>(b) - sum) / lower[i][i];
					}
				}

				// upper triangular
				for (int i = (int)size() - 1; i >= 0; i--) {
					for (int j = (int)size() - 1; j >= i; j--) {
						T sum = static_cast<T>(0);
						for (int k = (int)size() - 1; k > i; k--)
							sum += upper[i][k] * inverseUpper[k][j];

						float b = 0.0f;
						if (i == j)
							b = 1.0f;

						inverseUpper[i][j] = (b - sum) / upper[i][i];
					}
				}

				//return inverseLower;
				inv = inverseUpper * inverseLower;
			}
			else {
				inv = *this;
			}

			return inv;
		}
	};


	// row vector multiplication with matrix
	template<typename T>
	VectorN<T> VectorN<T>::operator*(const MatrixN<T>& m1) const {
		VectorN<T> result(size());

		if constexpr (std::is_arithmetic<T>::value) {
			for (size_t i = 0; i < size(); i++) {
				for (size_t j = 0; j < size(); j++) {
					result[i] += (*this)[j] * m1[j][i];
				}
			}
		}

		return result;
	}

	// v * v.Transpose()
	template<typename T>
	MatrixN<T> VectorN<T>::ExpandToMatrix() {
		MatrixN<T> result(size());

		if constexpr (std::is_arithmetic<T>::value) {
			for (size_t i = 0; i < size(); i++)
				for (size_t j = 0; j < size(); j++)
					result[i][j] += (*this)[i] * (*this)[j];
		}

		return result;
	}
}

#endif
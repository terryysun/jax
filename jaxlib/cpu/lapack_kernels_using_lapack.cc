/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/cpu/lapack_kernels.h"

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

namespace ffi = xla::ffi;

extern "C" {

jax::TriMatrixEquationSolver<ffi::DataType::F32>::FnType strsm_;
jax::TriMatrixEquationSolver<ffi::DataType::F64>::FnType dtrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C64>::FnType ctrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C128>::FnType ztrsm_;

jax::LuDecomposition<ffi::DataType::F32>::FnType sgetrf_;
jax::LuDecomposition<ffi::DataType::F64>::FnType dgetrf_;
jax::LuDecomposition<ffi::DataType::C64>::FnType cgetrf_;
jax::LuDecomposition<ffi::DataType::C128>::FnType zgetrf_;

jax::QrFactorization<ffi::DataType::F32>::FnType sgeqrf_;
jax::QrFactorization<ffi::DataType::F64>::FnType dgeqrf_;
jax::QrFactorization<ffi::DataType::C64>::FnType cgeqrf_;
jax::QrFactorization<ffi::DataType::C128>::FnType zgeqrf_;

jax::PivotingQrFactorization<ffi::DataType::F32>::FnType sgeqp3_;
jax::PivotingQrFactorization<ffi::DataType::F64>::FnType dgeqp3_;
jax::PivotingQrFactorization<ffi::DataType::C64>::FnType cgeqp3_;
jax::PivotingQrFactorization<ffi::DataType::C128>::FnType zgeqp3_;

jax::OrthogonalQr<ffi::DataType::F32>::FnType sorgqr_;
jax::OrthogonalQr<ffi::DataType::F64>::FnType dorgqr_;
jax::OrthogonalQr<ffi::DataType::C64>::FnType cungqr_;
jax::OrthogonalQr<ffi::DataType::C128>::FnType zungqr_;

jax::CholeskyFactorization<ffi::DataType::F32>::FnType spotrf_;
jax::CholeskyFactorization<ffi::DataType::F64>::FnType dpotrf_;
jax::CholeskyFactorization<ffi::DataType::C64>::FnType cpotrf_;
jax::CholeskyFactorization<ffi::DataType::C128>::FnType zpotrf_;

jax::SingularValueDecomposition<ffi::DataType::F32>::FnType sgesdd_;
jax::SingularValueDecomposition<ffi::DataType::F64>::FnType dgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C64>::FnType cgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C128>::FnType zgesdd_;

jax::SingularValueDecompositionQR<ffi::DataType::F32>::FnType sgesvd_;
jax::SingularValueDecompositionQR<ffi::DataType::F64>::FnType dgesvd_;
jax::SingularValueDecompositionQRComplex<ffi::DataType::C64>::FnType cgesvd_;
jax::SingularValueDecompositionQRComplex<ffi::DataType::C128>::FnType zgesvd_;

jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::FnType ssyevd_;
jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::FnType dsyevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C64>::FnType cheevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C128>::FnType zheevd_;

jax::EigenvalueDecomposition<ffi::DataType::F32>::FnType sgeev_;
jax::EigenvalueDecomposition<ffi::DataType::F64>::FnType dgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C64>::FnType cgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C128>::FnType zgeev_;

jax::SchurDecomposition<ffi::DataType::F32>::FnType sgees_;
jax::SchurDecomposition<ffi::DataType::F64>::FnType dgees_;
jax::SchurDecompositionComplex<ffi::DataType::C64>::FnType cgees_;
jax::SchurDecompositionComplex<ffi::DataType::C128>::FnType zgees_;

jax::HessenbergDecomposition<ffi::DataType::F32>::FnType sgehrd_;
jax::HessenbergDecomposition<ffi::DataType::F64>::FnType dgehrd_;
jax::HessenbergDecomposition<ffi::DataType::C64>::FnType cgehrd_;
jax::HessenbergDecomposition<ffi::DataType::C128>::FnType zgehrd_;

jax::TridiagonalReduction<ffi::DataType::F32>::FnType ssytrd_;
jax::TridiagonalReduction<ffi::DataType::F64>::FnType dsytrd_;
jax::TridiagonalReduction<ffi::DataType::C64>::FnType chetrd_;
jax::TridiagonalReduction<ffi::DataType::C128>::FnType zhetrd_;

jax::TridiagonalSolver<ffi::DataType::F32>::FnType sgtsv_;
jax::TridiagonalSolver<ffi::DataType::F64>::FnType dgtsv_;
jax::TridiagonalSolver<ffi::DataType::C64>::FnType cgtsv_;
jax::TridiagonalSolver<ffi::DataType::C128>::FnType zgtsv_;

}  // extern "C"

namespace jax {

static auto init = []() -> int {
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F32>>(strsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F64>>(dtrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C64>>(ctrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C128>>(ztrsm_);

  AssignKernelFn<LuDecomposition<ffi::DataType::F32>>(sgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::F64>>(dgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C64>>(cgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C128>>(zgetrf_);

  AssignKernelFn<QrFactorization<ffi::DataType::F32>>(sgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::F64>>(dgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C64>>(cgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C128>>(zgeqrf_);

  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F32>>(sgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F64>>(dgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C64>>(cgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C128>>(zgeqp3_);

  AssignKernelFn<OrthogonalQr<ffi::DataType::F32>>(sorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::F64>>(dorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C64>>(cungqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C128>>(zungqr_);

  AssignKernelFn<CholeskyFactorization<ffi::DataType::F32>>(spotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::F64>>(dpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C64>>(cpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C128>>(zpotrf_);

  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F32>>(sgesdd_);
  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F64>>(dgesdd_);
  AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C64>>(
      cgesdd_);
  AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C128>>(
      zgesdd_);

  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F32>>(sgesvd_);
  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F64>>(dgesvd_);
  AssignKernelFn<SingularValueDecompositionQRComplex<ffi::DataType::C64>>(
      cgesvd_);
  AssignKernelFn<SingularValueDecompositionQRComplex<ffi::DataType::C128>>(
      zgesvd_);

  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F32>>(ssyevd_);
  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F64>>(dsyevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C64>>(cheevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C128>>(
      zheevd_);

  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F32>>(sgeev_);
  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F64>>(dgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C64>>(cgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C128>>(zgeev_);

  AssignKernelFn<TridiagonalReduction<ffi::DataType::F32>>(ssytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::F64>>(dsytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C64>>(chetrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C128>>(zhetrd_);

  AssignKernelFn<SchurDecomposition<ffi::DataType::F32>>(sgees_);
  AssignKernelFn<SchurDecomposition<ffi::DataType::F64>>(dgees_);
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C64>>(cgees_);
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C128>>(zgees_);

  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F32>>(sgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F64>>(dgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C64>>(cgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C128>>(zgehrd_);

  AssignKernelFn<TridiagonalSolver<ffi::DataType::F32>>(sgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::F64>>(dgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C64>>(cgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C128>>(zgtsv_);

  return 0;
}();

}  // namespace jax

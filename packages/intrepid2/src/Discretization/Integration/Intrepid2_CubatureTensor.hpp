// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid2_CubatureTensor.hpp
    \brief  Header file for the Intrepid2::CubatureTensor class.
    \author Created by P. Bochev and D. Ridzal.
            Kokkorized by Kyungjoo Kim
*/

#ifndef __INTREPID2_CUBATURE_TENSOR_HPP__
#define __INTREPID2_CUBATURE_TENSOR_HPP__

#include "Intrepid2_ConfigDefs.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Intrepid2_CubatureDirect.hpp"

namespace Intrepid2 {

  /** \class Intrepid2::CubatureTensor
      \brief Defines tensor-product cubature (integration) rules in Intrepid.
  */
  template<typename DeviceType = void,
           typename pointValueType = double,
           typename weightValueType = double>
  class CubatureTensor
    : public Cubature<DeviceType,pointValueType,weightValueType> {
  private:

    /** \brief Array of cubature rules.
     */
    ordinal_type numCubatures_;

    CubatureDirect<DeviceType,pointValueType,weightValueType> cubatures_[Parameters::MaxTensorComponents];

    /** \brief Dimension of integration domain.
     */
    ordinal_type dimension_;

  public:

    template<typename cubPointValueType,  class ...cubPointProperties,
             typename cubWeightValueType, class ...cubWeightProperties>
    void
    getCubatureImpl( Kokkos::DynRankView<cubPointValueType, cubPointProperties...>  cubPoints,
                     Kokkos::DynRankView<cubWeightValueType,cubWeightProperties...> cubWeights ) const;

    using PointViewType             = typename Cubature<DeviceType,pointValueType,weightValueType>::PointViewType;
    using weightViewType            = typename Cubature<DeviceType,pointValueType,weightValueType>::weightViewType;

    /// KK: following should be updated with nate's tensor work
    using PointViewTypeAllocatable  = typename Cubature<DeviceType,pointValueType,weightValueType>::PointViewTypeAllocatable;
    using WeightViewTypeAllocatable = typename Cubature<DeviceType,pointValueType,weightValueType>::WeightViewTypeAllocatable;
    using TensorPointDataType       = typename Cubature<DeviceType,pointValueType,weightValueType>::TensorPointDataType;
    using TensorWeightDataType      = typename Cubature<DeviceType,pointValueType,weightValueType>::TensorWeightDataType;

    using Cubature<DeviceType,pointValueType,weightValueType>::getCubature;

    virtual
    void
    getCubature( PointViewType  cubPoints,
                 weightViewType cubWeights ) const override {
      getCubatureImpl( cubPoints,
                       cubWeights );
    }
      
    /** \brief Returns a points container appropriate for passing to getCubature().

        \return cubPoints  - Data structure sized for the cubature points.
    */
    virtual TensorPointDataType allocateCubaturePoints() const override
    {
      std::vector< PointViewTypeAllocatable > cubaturePointComponents(numCubatures_);
      
      for (ordinal_type i=0;i<numCubatures_;++i)
      {
        cubaturePointComponents[i] = PointViewTypeAllocatable("cubature points", cubatures_[i].getNumPoints(), cubatures_[i].getDimension());
      }
      
      return TensorPointDataType(cubaturePointComponents);
    }
    
    /** \brief Returns a weight container appropriate for passing to getCubature().

        \return cubWeights  - Data structure sized for the cubature weights.
    */
    virtual TensorWeightDataType allocateCubatureWeights() const override
    {
      /// KK: this needs to be updated with nate tensor work      
      using WeightDataType = Data<weightValueType,typename DeviceType::execution_space>;
      
      std::vector< WeightDataType > cubatureWeightComponents(numCubatures_);
      for (ordinal_type i=0;i<numCubatures_;++i)
      {
        cubatureWeightComponents[i] = WeightDataType(WeightViewTypeAllocatable("cubature weights", cubatures_[i].getNumPoints()));
      }
      
      return TensorWeightDataType(cubatureWeightComponents);
    }
    
    /** \brief Returns tensor cubature points and weights.  For non-tensor cubatures, the tensor structures are trivial, thin wrappers around the data returned by getCubature().  The provided containers should be pre-allocated through calls to allocateCubaturePoints() and allocateCubatureWeights().

        \param cubPoints       [out]   - TensorPoints structure containing the cubature points.
        \param cubWeights      [out]  - TensorData structure containing cubature weights.
    */
    virtual
    void
    getCubature( const TensorPointDataType & tensorCubPoints,
                 const TensorWeightDataType & tensorCubWeights) const override {
      for (ordinal_type i=0;i<numCubatures_;++i)
      {
        cubatures_[i].getCubature(tensorCubPoints.getTensorComponent(i), tensorCubWeights.getTensorComponent(i).getUnderlyingView());
      }
    }

    /** \brief Returns the number of cubature points.
     */
    virtual
    ordinal_type
    getNumPoints() const override {
      ordinal_type numCubPoints = 1;
      for (ordinal_type i=0;i<numCubatures_;++i)
        numCubPoints *= cubatures_[i].getNumPoints();
      return numCubPoints;
    }

    /** \brief Returns dimension of integration domain.
     */
    virtual
    ordinal_type
    getDimension() const override {
      return dimension_;
    }

    /** \brief Returns cubature name.
     */
    virtual
    const char*
    getName() const override {
      return "CubatureTensor";
    }

    virtual
    ordinal_type 
    getAccuracy() const override {
      ordinal_type r_val = 0;
      for (ordinal_type i=0;i<numCubatures_;++i)
        r_val = Util<ordinal_type>::max(r_val, cubatures_[i].getAccuracy());
      return r_val;
    }

    /** \brief Return the number of cubatures.
     */
    ordinal_type getNumCubatures() const {
      return numCubatures_;
    }

    /** \brief Returns max. degree of polynomials that are integrated exactly.
     */
    void getAccuracy( ordinal_type *accuracy ) const {
      for (ordinal_type i=0;i<numCubatures_;++i)
        accuracy[i] = cubatures_[i].getAccuracy();
    }

    CubatureTensor() 
      : numCubatures_(0),
        dimension_(0) {}

    CubatureTensor(const CubatureTensor &b)
      : numCubatures_(b.numCubatures_),
        dimension_(b.dimension_) {
      for (ordinal_type i=0;i<numCubatures_;++i) 
        cubatures_[i] = b.cubatures_[i];
    }

    /** \brief Constructor.

        \param cubature1        [in]     - First direct cubature rule.
        \param cubature2        [in]     - Second direct cubature rule.
    */
    template<typename CubatureType0,
             typename CubatureType1>
    CubatureTensor( const CubatureType0 cubature0,
                    const CubatureType1 cubature1 )
      : numCubatures_(2),
        dimension_(cubature0.getDimension()+cubature1.getDimension()) {
      cubatures_[0] = cubature0;
      cubatures_[1] = cubature1;
    }

    /** \brief Constructor.

        \param cubature1        [in]     - First direct cubature rule.
        \param cubature2        [in]     - Second direct cubature rule.
        \param cubature3        [in]     - Third direct cubature rule.
    */
    template<typename CubatureType0,
             typename CubatureType1,
             typename CubatureType2>
    CubatureTensor( const CubatureType0 cubature0,
                    const CubatureType1 cubature1,
                    const CubatureType2 cubature2 ) 
      : numCubatures_(3),
        dimension_(cubature0.getDimension()+cubature1.getDimension()+cubature2.getDimension()) {
      cubatures_[0] = cubature0;
      cubatures_[1] = cubature1;
      cubatures_[2] = cubature2;
    }

    /** \brief Constructor for extending an existing CubatureTensor object with an additional direct cubature rule.

        \param cubatureTensor    [in] - Existing CubatureTensor object.
        \param cubatureExtension [in] - The direct cubature rule to use to extend in the new dimension.
    */
    template<typename DirectCubature>
    CubatureTensor( const CubatureTensor cubatureTensor,
                    const DirectCubature cubatureExtension )
      : numCubatures_(cubatureTensor.getNumCubatures()+1),
        dimension_(cubatureTensor.getDimension()+cubatureExtension.getDimension()) {
      for (ordinal_type i=0;i<cubatureTensor.getNumCubatures();++i)
        cubatures_[i] = cubatureTensor.cubatures_[i];
      cubatures_[cubatureTensor.getNumCubatures()] = cubatureExtension;
    }
  };


} // end namespace Intrepid2


// include templated definitions
#include <Intrepid2_CubatureTensorDef.hpp>

#endif

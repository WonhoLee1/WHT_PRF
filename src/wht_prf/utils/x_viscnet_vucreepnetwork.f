      subroutine vucreepnetwork(
c Read only -
     *   nblock, networkID, 
     *   nstatev, nfieldv, nprops, nDg,
     *   stepTime, totalTime, dt, 
     *   jGlobalElemUid, kIntSta, kSegment, kmp, 
     *   cmname, props, coordMp, 
     *   tempOld, fieldOld, stateOld, 
     *   tempNew, fieldNew, 
     *   nIarray, i_array,
     *   nRarray, r_array,
     *   q, p, eqcs, TrCc, 
c Write only -
     *   dg, stateNew )
c
      include 'vaba_param.inc'
c
      dimension jGlobalElemUid(nblock),
     *     props(nprops), 
     *     tempOld(nblock),
     *     fieldOld(nblock,nfieldv), 
     *     stateOld(nblock,nstatev), 
     *     tempNew(nblock),
     *     fieldNew(nblock,nfieldv),
     *     q(nblock), p(nblock), 
     *     eqcs(nblock), 
     *     trCc(nblock),
     *     dg(nblock,nDg),
     *     stateNew(nblock,nstatev)
      dimension r_array(nblock,nRarray)
      dimension i_array(nblock,nIarray)
*
      character*80 cmname
*
      parameter ( one = 1.d0, half = 0.5d0 )
      parameter ( eqcsSmall = 1.d-8 )
      parameter ( rMinVal = 1.d-12 )
*
      parameter ( l_tmp_g    = 1,
     *     l_tmp_DgDq      = 2,
     *     l_tmp_DgDeqcs   = 3,
     *     l_tmp_DgDi1c    = 4 )
*
      if ( networkID .eq. 1 ) then
*     LAW=STRAIN
         rA = props(1)
         rN = props(2)
         rM = props(3)
         do k = 1, nblock
            om1 = one / ( one + rM )
            test = half - sign( half, q(k) - rMinVal )
            qInv = ( one - test ) / ( q(k) + test )
            eqcs_t = eqcs(k)
            if ( eqcs_t .le. eqcsSmall .and. q(k).gt.rMinVal ) then 
* Initial guess based on constant creep strain rate during increment
               eqcs_t = dt*(exp(log(rA)+rN*log(q(k)))*
     *              ((one+rM)*dt)**rM)
            end if
*
            test2 = half - sign( half, eqcs_t - rMinVal )
            eqcsInv = ( one - test2 ) / ( eqcs_t + test2 )
            g = dt*( rA*(q(k)**rN) *
     *           ((one+rM)*(test2+eqcs_t))**rM 
     *           )**om1
            dg(k,l_tmp_g) = g
            dg(k,l_tmp_dgdq) = qInv * rN * om1 * g
            dg(k,l_tmp_dgdeqcs) = eqcsInv * rM * om1 * g
         end do
      else if ( networkID .eq. 2 ) then
*     LAW=HYPERB
         A = props(1)
         B = props(2)
         rN = props(3)
         do k = 1, nblock
            t1=exp(B*q(k))
            t2=exp(-B*q(k))
            sinh = half*(t1-t2)
            cosh = half*(t1-t2)
            gtmp = dt*A*(sinh)**(rN-one)
            g = sinh * gtmp 
            dgdq = B * rN * cosh * gtmp
            dg(k,l_tmp_g) = g
            dg(k,l_tmp_dgdq) = dgdq
         end do
      end if
*
      if ( nstatev .ge. 3 ) then
         do k = 1, nblock
            stateNew(k,1) = r_array(k,1)
            stateNew(k,2) = r_array(k,2)
            stateNew(k,3) = r_array(k,3)
         end do
      end if
*
      return
      end



   subroutine zblock_lu1( id, a, lda, na, k )
!  ===================================================================
!  does a partitioning of a matrix and a partial inversion to
!  get the upper subblock of the inverse
!
!  a      -- complex*16 of (lda,size) the matrix to be inverted where
!            size is the sum of the all elements of blk_sz on return,
!            the upper subblock of a is untouched, and postprocessing
!            is needed to obtain the inverse
!
!  blk_sz -- integer of (nblk), the size of each subblock of matrix a
!
!  nblk   -- number of blocks (number of elements in blk_sz)
!
!  ipvt   -- integer of (mp), work array
!
!  idcol  -- integer of (blk_sz(1)), if idcol(i)=idcol(j), then
!            the two columns are equivalent by symmetry, and only
!            the min(i,j) th column of the inverse is calculated.
!
!  k      -- returns actual number of columns in the calculated inverse
!  ===================================================================
!
   use MatrixBlockInversionModule
   use KindParamModule

   implicit none

!
   integer (kind=IntKind), intent(in) ::  id, lda, na
   integer (kind=IntKind), intent(out) :: k
!
   complex (kind=CmplxKind), target :: a(lda,na)
!
   integer (kind=IntKind) :: i, m, n, info, blk1, nblk
   integer (kind=IntKind) :: ioff, joff, iblk
   integer (kind=IntKind), pointer :: p_ipvt(:)
   integer (kind=IntKind), pointer :: p_idcol(:)
!
!  print *, "start zblock_lu1"
!  ===================================================================
!  eliminate columns that are equiv due to symmetry
!  ===================================================================
!

   blk1 = MatrixBlockSizes(1,id)
   nblk = NumBlocks(id)
   p_idcol => idcol(1:blk1,id)
   k = blk1+1
!  print *,'k = ',k,', blk1 = ',blk1,', lda = ',lda,', na = ',na,', nblk = ',nblk
   do i = blk1,1,-1
      if ( p_idcol(1)==0 .or. p_idcol(i)==i ) then
         k = k-1
         if ( k/=i ) then
            call zcopy( na-blk1, a(blk1+1,i), 1, a(blk1+1,k), 1 )
         endif
      endif
   enddo
!
!  print *, 'nblk = ',nblk
   if ( isLU ) then
!     ================================================================
!     Do block LU
!     ================================================================
      n = MatrixBlockSizes(nblk,id)
      joff = na-n
      do iblk = nblk,2,-1
         m = n
         ioff = joff
         n = MatrixBlockSizes(iblk-1,id)
         joff = joff-n
!        =============================================================
!        invert the diagonal blk_sz(iblk) x blk_sz(iblk) block
!        =============================================================
         p_ipvt => ipvt(1:m)
!        -------------------------------------------------------------
         call zgetrf( m,m,a(ioff+1,ioff+1),lda,p_ipvt,info )
         if (info /= 0) then
            call ErrorHandler('BinvMatrix','zgetrf failed', info)
         endif
!        -------------------------------------------------------------
!        =============================================================
!        calculate the inverse of above multiplying the row block
!        blk_sz(iblk) x ioff
!        =============================================================
!        -------------------------------------------------------------
         call zgetrs( 'n',m,ioff,a(ioff+1,ioff+1),lda,p_ipvt,          &
                       a(ioff+1,1),lda,info )
         if (info /= 0) then
            call ErrorHandler('BinvMatrix','zgetrs failed', info)
         endif
         if (iblk.gt.2) then
!           ----------------------------------------------------------
            call zgemm( 'n','n',n,ioff-k+1,na-ioff,-cone,              &
                        a(joff+1,ioff+1),lda,a(ioff+1,k),lda,          &
                        cone,a(joff+1,k),lda )
!           ----------------------------------------------------------
            call zgemm( 'n','n',joff,n,na-ioff,-cone,a(1,ioff+1),lda,  &
                        a(ioff+1,joff+1),lda,cone,a(1,joff+1),lda )
!           ----------------------------------------------------------
         endif
      enddo
!     ----------------------------------------------------------------
      call zgemm( 'n', 'n', blk1, blk1-k+1, na-blk1, -cone,            &
                a(1,blk1+1), lda, a(blk1+1,k), lda, cone, a, lda )
!     ----------------------------------------------------------------
   endif
!
   k = blk1-k+1
!
   end subroutine zblock_lu1

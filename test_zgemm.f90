PROGRAM blas_test

  implicit none
  
  INTEGER :: M, N, K
  complex*16, ALLOCATABLE :: A(:,:), B(:,:), C(:,:)
  INTEGER :: i, Niter
  REAL*8 :: dtime, mysecond

  REAL*8, ALLOCATABLE :: realPartA(:,:), imagPartA(:,:)
  REAL*8, ALLOCATABLE :: realPartB(:,:), imagPartB(:,:)

  
  Niter=10

  ! Read matrix dimensions from user input
  READ(*,*) M, N, K
  
  ! Allocate memory for the matrices
  ALLOCATE(A(N, K), B(K, M), C(N, M))
  ALLOCATE(realPartA(M, K), imagPartA(M, K), realPartB(K, N), imagPartB(K, N))

  
  ! Initialize matrices A and B (for example)
! A = 1.0d0
! B = 2.0d0
  call random_number(realPartA)
  call random_number(imagPartA)
  A = cmplx(realPartA, imagPartA)
  call random_number(realPartB)
  call random_number(imagPartB)
  B = cmplx(realPartB, imagPartB)
  
  dtime = mysecond()
  do i=1, Niter
    CALL ZGEMM('N', 'N', N, M, K, (1.0d0,0.0d0), A, N, B, K, (0.0d0,0.0d0), C, N)
  end do
  dtime = mysecond() - dtime
  dtime = dtime / Niter
  
  write(*,'(A F20.6)') "runtime(s):", dtime
  
  ! Display the result
  !WRITE(*,*) "Resultant Matrix C:"
  !DO i = 1, N
  !  WRITE(*, '(100F8.2)') (C(i, j), j = 1, M)
  !ENDDO
  
  
  ! Deallocate memory
  DEALLOCATE(A, B, C)
  DEALLOCATE(realPartA, imagPartA, realPartB, imagPartB)
  
END PROGRAM blas_test


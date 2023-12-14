PROGRAM blas_test

  implicit none
  
  INTEGER :: M, N, K
  REAL*8, ALLOCATABLE :: A(:,:), B(:,:), C(:,:)
  INTEGER :: i, Niter, j
  REAL*8 :: dtime, mysecond
  REAL*8 :: check
  
  ! Read matrix dimensions from user input
  READ(*,*) M, N, K, Niter
  
  ! Allocate memory for the matrices
  ALLOCATE(A(M, K), B(K, N), C(M, N))
  
  ! Initialize matrices A and B (for example)
  A = 2.0d0
  B = 0.5d0
! call random_number(A)
! call random_number(B)
  
  dtime = mysecond()
  do i=1, Niter
!    print *, "iter=",i
    CALL DGEMM('N', 'N', M, N, K, 1.0d0, A, M, B, K, 0.0d0, C, M)
  end do
  dtime = mysecond() - dtime
  dtime = dtime / Niter

!$omp parallel do reduction(+:check) private(j)
  do i=1, M
    do j=1, N
      check=check+C(i,j)
    enddo
  enddo 
!$omp end parallel do
  check=check/M/N
  
  write(*,'(A F20.6)') "runtime(s):", dtime
  
  ! Display the result
  !WRITE(*,*) "Resultant Matrix C:"
  !DO i = 1, M
  !  WRITE(*, '(100F8.2)') (C(i, j), j = 1, N)
  !ENDDO
  WRITE(*,"(A F15.2 I10)") "# Result check", check, K
  
  
  ! Deallocate memory
  DEALLOCATE(A, B, C)
  
END PROGRAM blas_test

